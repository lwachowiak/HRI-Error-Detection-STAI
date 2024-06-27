### OWN CODE IMPORTS ###
from get_metrics import get_metrics
from data_loader import DataLoader_HRI

### ML IMPORTS ###
from tsai.data.all import *
from tsai.models.utils import *
from tsai.models import MINIROCKET, MINIROCKET_Pytorch
from tsai.all import my_setup, accuracy, F1Score, CrossEntropyLossFlat, FocalLossFlat, Learner, TST, LSTM_FCN, TransformerLSTMPlus, HydraMultiRocketPlus, ConvTranPlus
from fastai.callback.all import EarlyStoppingCallback
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch

### TRACKING IMPORTS ###
import optuna
import optuna.study.study
from optuna.integration import WeightsAndBiasesCallback
import wandb

### OTHER IMPORTS ###
import json
import datetime
import platform
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import pickle

# TODO:
# Models:
#   - Try annotation/outlier processing: https://www.sktime.net/en/stable/api_reference/annotation.html
#   - Try prediction/forcasting --> unlikely sequence --> outlier --> label 1
#   - Foundation Models


class TS_Model_Trainer:
    """
    A class for training individual models, doing optuna hyperparamter search, and inference with already trained models. 
    """

    def __init__(self, folder: str, n_jobs: int, config_name: str):
        '''
        Initialize the TS_Model_Trainer with the data folder and the task to be trained on.
        :param data_folder: The folder containing the data.
        :param n_jobs: CPU usage parameter
        '''
        self.folder = folder
        self.data = DataLoader_HRI(folder+"data/")
        self.task = None
        self.TASK_TO_COLUMN = {0: "UserAwkwardness",
                               1: "RobotMistake", 2: "InteractionRupture"}
        self.n_jobs = n_jobs
        self.objective_per_model = {
            "MiniRocket": self.optuna_objective_classic,
            "RandomForest": self.optuna_objective_classic,
            "XGBoost": self.optuna_objective_classic,
            "MiniRocketTorch": self.optuna_objective_tsai,
            "TST": self.optuna_objective_tsai,
            "LSTM_FCN": self.optuna_objective_tsai,
            "ConvTranPlus": self.optuna_objective_tsai,
            "TransformerLSTMPlus": self.optuna_objective_tsai
        }
        self.config = self.read_config(folder+"code/"+config_name)
        # TODO change this to a separate function that just splits based on comma
        self.column_removal_dict = {"REMOVE_NOTHING": ["REMOVE_NOTHING"],
                                    "opensmile": ["opensmile"],
                                    "speaker": ["speaker"],
                                    "openpose": ["openpose"],
                                    "openface": ["openface"],
                                    "openpose, speaker": ["openpose", "speaker"],
                                    "speaker, openpose, openface": ["speaker", "openpose", "openface"],
                                    "speaker, openface, opensmile": ["speaker", "openface", "opensmile"],
                                    "c_openface": ["c_openface"],
                                    "openpose, c_openface": ["openpose", "c_openface"],
                                    "vel_dist": ["vel_dist"],
                                    "vel_dist, c_openface": ["vel_dist", "c_openface"]
                                    }
        self.loss_dict = {"CrossEntropyLossFlat": CrossEntropyLossFlat(),
                          "FocalLossFlat": FocalLossFlat()}

    def read_config(self, file_path: str) -> dict:
        """Reads a JSON configuration file and returns the configuration as a dictionary."""
        try:
            with open(file_path, 'r') as file:
                config = json.load(file)
                # print config
                print("\nConfiguration loaded successfully.")
                for key, value in config.items():
                    print(f"{key}: {value}")
                self.task = config["task"]
                return config
        except FileNotFoundError:
            print("\nError: The configuration", file_path, "was not found.")
        except json.JSONDecodeError:
            print("\nError: The configuration file is not in proper JSON format.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_full_test_preds(self, model: object, val_X_TS_list: list, interval_length: int, stride_eval: int, model_type: str, batch_tfms: list = None, start_padding: bool = False) -> list:
        '''Get full test predictions by repeating the predictions based on interval_length and stride_eval.
        :param model: The model to evaluate.
        :param val_X_TS_list: List of validation/test data per session.
        :param interval_length: The length of the interval to predict.
        :param stride_eval: The stride to evaluate the model.
        :param model_type: Either "Classic" or "TSAI", which have different API calls
        :param batch_tfms: List of batch transformations to apply, if any
        '''
        if model_type not in ["Classic", "TSAI"]:
            raise ValueError(
                "Model type not supported. Parameter model_type must be either 'Classic' or 'TSAI'.")
        test_preds = []
        for val_X_TS in val_X_TS_list:  # per session
            if model_type == "Classic":
                pred = model.predict(val_X_TS)
            elif model_type == "TSAI":
                for tfm in batch_tfms:
                    val_X_TS = tfm(val_X_TS)
                valid_probas, valid_targets = model.get_X_preds(
                    X=val_X_TS, y=None, with_decoded=False)  # don't use the automatic decoding, there is a bug in the tsai library
                pred = [model.dls.vocab[p]
                        for p in np.argmax(valid_probas, axis=1)]
            # for each sample in the session, repeat the prediction based on interval_length and stride_eval
            processed_preds = []
            for i, pr in enumerate(pred):
                if i == 0 and not start_padding:
                    # first prediction, so append it interval_length times. if start_padding was used, the first prediction only pertains to stride elements
                    processed_preds.extend([pr]*interval_length)
                else:
                    # all other predictions are appended stride_eval times
                    processed_preds.extend([pr]*stride_eval)
            test_preds.append(processed_preds)
        return test_preds

    def get_eval_metrics(self, preds: list, dataset="val", verbose=False) -> dict:
        '''Evaluate model on self.data.val_X and self.data.val_Y. The final missing values in preds are filled with 0s.
        :param preds: List of predictions per session. Per session there is one list of prediction labels.
        :param dataset: The dataset to evaluate on (val or test)
        :output eval_scores: Dictionary containing the evaluation scores (accuracy, macro f1, macro precision, macro recall)
        '''
        y_true = []
        # iterate over all preds (per session) and append 0s if necessary
        session_ids = self.data.val_Y["session"].unique()
        print("Session IDs Val:", session_ids)
        # TODO is preds in the same order as session_ids????
        for i, session_id in enumerate(session_ids):
            y_true.append(
                self.data.val_Y[self.data.val_Y['session'] == session_id][self.TASK_TO_COLUMN[self.task]].values)
            # append 0s to preds if necessary
            if len(preds[i]) < len(self.data.val_Y[self.data.val_Y['session'] == session_id]):
                to_append = len(
                    self.data.val_Y[self.data.val_Y['session'] == session_id])-len(preds[i])
                preds[i] = np.append(preds[i], [0]*(to_append))
                if verbose:
                    print("Appended", to_append,
                          "0s to preds for session", session_id, "resulting in", len(preds[i]), "predictions")
            if (len(preds[i]) != len(self.data.val_Y[self.data.val_Y['session'] == session_id])):
                print("ERROR: Length of preds and val_Y do not match for session", session_id,
                      "preds:", len(preds[i]), "val_Y:", len(self.data.val_Y[self.data.val_Y['session'] == session_id]))
        # flatten preds
        preds = np.concatenate(preds)
        y_true = np.concatenate(y_true)
        print("Final preds length", len(preds), len(y_true))

        eval_scores = get_metrics(y_pred=preds, y_true=y_true, tolerance=50)

        # print confusion matrix
        if verbose:
            print(confusion_matrix(y_true, preds, normalize='all'))
            print(eval_scores)

        return eval_scores

    def data_from_config(self, data_values: dict, format: str, columns_to_remove: list, fold: int) -> tuple:
        """
        create the datasets for training based on the configuration and the trial parameters.
        params: data_values: dict: The data values to use for the data creation.
        params: format: str: The format of the data to return. Either "timeseries" or "classic".
        params: columns_to_remove: list: The columns to remove from the data.
        params: fold: int: The fold to use for validation data
        output: tuple: Tuple containing the validation and training datasets.
        """
        if fold not in range(1, 5):
            print("Warning: Training on all data, including validation set.")
        # TODO change variable names from TS to sth generic
        if format == "classic":
            # get summary format for classic models
            val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order = self.data.get_summary_format(
                interval_length=data_values["interval_length"],
                stride_train=data_values["stride_train"],
                stride_eval=data_values["stride_eval"],
                fps=data_values["fps"],
                label_creation=data_values["label_creation"],
                summary=data_values["summary"],
                oversampling_rate=data_values["oversampling_rate"],
                undersampling_rate=data_values["undersampling_rate"],
                task=self.task,
                fold=fold,
                rescaling=data_values["rescaling"],
                start_paddding=data_values["start_padding"]

            )

        if format == "timeseries":
            # get timeseries format
            val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order = self.data.get_timeseries_format(
                interval_length=data_values["interval_length"],
                stride_train=data_values["stride_train"],
                stride_eval=data_values["stride_eval"], verbose=False,
                fps=data_values["fps"],
                label_creation=data_values["label_creation"],
                oversampling_rate=data_values["oversampling_rate"],
                undersampling_rate=data_values["undersampling_rate"],
                task=self.task,
                fold=fold,
                rescaling=data_values["rescaling"],
                start_padding=data_values["start_padding"]
            )

        # nan handling
        if data_values["nan_handling"] == "zeros":
            train_X_TS = np.nan_to_num(train_X_TS, nan=0)
            val_X_TS_list = [np.nan_to_num(val_X_TS, nan=0)
                             for val_X_TS in val_X_TS_list]
        if data_values["nan_handling"] == "avg":
            train_X_TS = DataLoader_HRI.impute_nan_with_feature_mean(
                train_X_TS)
            val_X_TS_list = [DataLoader_HRI.impute_nan_with_feature_mean(
                val_X_TS) for val_X_TS in val_X_TS_list]
        # feature removal
        train_X_TS, new_column_order = self.remove_columns(columns_to_remove=columns_to_remove,
                                                           data_X=train_X_TS, column_order=column_order)
        val_X_TS_list, new_column_order = self.remove_columns(columns_to_remove=columns_to_remove,
                                                              data_X=val_X_TS_list, column_order=column_order)

        train_Y_TS_task = train_Y_TS[:, self.task]

        return val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, new_column_order, train_Y_TS_task

    def get_data_values(self, trial: optuna.Trial) -> tuple:
        """ Get the data values for the trial based on the configuration and the trial parameters.
        :param trial: optuna.Trial: The trial object.
        :output data_values: dict: The data values to use for the data creation.
        :output columns_to_remove: list: The columns to remove from the data.
        """
        data_params = self.config["data_params"]
        data_values = {}
        data_values["interval_length"] = trial.suggest_int(
            "interval_length", low=data_params["interval_length"]["low"], high=data_params["interval_length"]["high"], step=data_params["interval_length"]["step"])
        # strides must be leq than interval_length
        data_values["stride_train"] = trial.suggest_int(
            "stride_train", low=data_params["stride_train"]["low"], high=min(data_values["interval_length"], data_params["stride_train"]["high"]), step=data_params["interval_length"]["step"])
        data_values["stride_eval"] = trial.suggest_int(
            "stride_eval", low=data_params["stride_eval"]["low"], high=min(data_values["interval_length"], data_params["stride_eval"]["high"]), step=data_params["interval_length"]["step"])
        data_values["fps"] = trial.suggest_categorical(
            "fps", data_params["fps"])
        data_values["columns_to_remove"] = trial.suggest_categorical("columns_to_remove",
                                                                     data_params["columns_to_remove"])
        columns_to_remove = self.column_removal_dict[data_values["columns_to_remove"]]
        data_values["label_creation"] = trial.suggest_categorical(
            "label_creation", data_params["label_creation"])
        data_values["nan_handling"] = trial.suggest_categorical(
            "nan_handling", data_params["nan_handling"])
        data_values["oversampling_rate"] = trial.suggest_float(
            "oversampling_rate", low=data_params["oversampling_rate"]["low"], high=data_params["oversampling_rate"]["high"], step=data_params["oversampling_rate"]["step"])
        data_values["undersampling_rate"] = trial.suggest_float(
            "undersampling_rate", low=data_params["undersampling_rate"]["low"], high=data_params["undersampling_rate"]["high"], step=data_params["undersampling_rate"]["step"])
        data_values["rescaling"] = trial.suggest_categorical(
            "rescaling", data_params["rescaling"])
        data_values["start_padding"] = trial.suggest_categorical(
            "start_padding", data_params["start_padding"])
        if "summary" in data_params:
            data_values["summary"] = trial.suggest_categorical(
                "summary", data_params["summary"])
        return data_values, columns_to_remove

    def merge_val_train(self, val_X_TS_list: list, val_Y_TS_list: list, train_X_TS: np.array, train_Y_TS_task: np.array) -> tuple:
        """
        Merge the training and all validation sets (per session) into one dataset so that the Torch models can be trained on it.
        """
        all_X = train_X_TS
        for val_X_TS in val_X_TS_list:
            # print(val_X_TS.shape, all_X.shape)
            all_X = np.concatenate((all_X, val_X_TS), axis=0)
        all_Y = train_Y_TS_task
        for val_Y_TS in val_Y_TS_list:
            val_Y_TS = val_Y_TS[:, self.task]
            all_Y = np.concatenate((all_Y, val_Y_TS), axis=0)
        # print(all_X.shape, all_Y.shape)
        splits = [range(0, len(train_X_TS)), range(
            len(train_X_TS), len(all_X))]
        return all_X, all_Y, splits

    def optuna_study(self, n_trials: int, model_type: str, study_name: str, verbose=False) -> optuna.study.Study:
        """Performs an Optuna study to optimize the hyperparameters of the model.
        :param n_trials: The number of search trials to perform.
        :param model_type: The type of model to optimize (MiniRocket, TST).
        """
        wandb_kwargs = {"project": "HRI-Errors",
                        "name": study_name+"_task_"+str(self.task), "group": model_type}
        wandbc = WeightsAndBiasesCallback(
            metric_name=["accuracy", "macro f1"], wandb_kwargs=wandb_kwargs)

        # storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(
            directions=["maximize", "maximize"], study_name=study_name)  # , storage=storage_name, load_if_exists=True)
        print(f"Sampler is {study.sampler.__class__.__name__}")

        objective = self.objective_per_model[model_type]
        study.optimize(objective, n_trials=n_trials, callbacks=[wandbc])

        trial_with_highest_accuracy = max(
            study.best_trials, key=lambda t: t.values[0])
        if verbose:
            print("Best trial Accuracy:")
            print(
                "Values: ", trial_with_highest_accuracy.values, "Params: ", trial_with_highest_accuracy.params)
            for key, value in trial_with_highest_accuracy.params.items():
                print(f"    {key}: {value}")

        # attach optuna visualization to wandb
        self.get_trials_figures(study, 0, "accuracy")
        # self.get_trials_figures(study, 1, "macro f1")

        # save best params to json
        best_params = {}
        best_params["task"] = self.task
        best_params["model_type"] = model_type
        best_params["data_params"] = {}
        best_params["model_params"] = {}
        best_params["stats"] = {}

        f = "best_{}".format
        for param_name, param_value in trial_with_highest_accuracy.params.items():
            wandb.run.summary[f(param_name)] = param_value
            if param_name in self.config["data_params"]:
                best_params["data_params"][param_name] = param_value
            elif param_name in self.config["model_params"]:
                best_params["model_params"][param_name] = param_value

        # for value_name, value in zip(study.directions, trial_with_highest_accuracy.values):
        #    wandb.run.summary[f(value_name)] = value
        #    best_params["stats"][value_name] = value

        wandb.run.summary["best accuracy"] = trial_with_highest_accuracy.values[0]
        wandb.run.summary["best macro f1"] = trial_with_highest_accuracy.values[1]
        best_params["stats"]["accuracy"] = trial_with_highest_accuracy.values[0]
        best_params["stats"]["macro f1"] = trial_with_highest_accuracy.values[1]

        wandb.finish()

        with open(self.folder+"code/best_model_configs/"+str(study_name)+".json", "w") as f:
            json.dump(best_params, f)
        # safe the best model overall to pkl (based on fold 4 for validation)
        self.train_and_save_best_model(
            model_config=str(study_name)+".json", name_extension="best_"+str(study_name))

        return study

    def get_trials_figures(self, study: optuna.study.Study, target_index: int, target_name: str) -> None:
        '''Get optuna visualization figures and log them to wandb. Summary visualizations for full search.
        params: study: optuna.study.Study: The optuna study object.
        params: target_index: int: The index of the target value to plot, 0 for accuracy, 1 for macro f1.
        params: target_name: str: The name of the target value to plot, used for the wandb log.
        '''
        # fig = optuna.visualization.plot_optimization_history(
        #    study, target=lambda t: t.values[target_index], target_name=target_name)
        # wandb.log({"optuna_optimization_history_"+target_name: fig})
        fig = optuna.visualization.plot_parallel_coordinate(
            study, target=lambda t: t.values[target_index], target_name=target_name)
        wandb.log({"optuna_parallel_coordinate_"+target_name: fig})
        fig = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[target_index], target_name=target_name)
        wandb.log({"optuna_param_importances_"+target_name: fig})
        fig = optuna.visualization.plot_slice(
            study, target=lambda t: t.values[target_index], target_name=target_name)
        wandb.log({"optuna_slice_"+target_name: fig})

    def remove_columns(self, columns_to_remove: list, data_X: np.array, column_order: list) -> tuple:
        '''Remove columns from the data.
        :param columns_to_remove: List of columns to remove.
        :param data_X: The data to remove the columns from. Either a list of np.arrays or a np.array.
        :param column_order: The order of the columns in the data.
        :output new_data_X: The data with the specified columns removed and the new column order.
        '''
        # depending on whether data_X is list or np.array
        if isinstance(data_X, list):  # val/test
            new_data_X = [val_X_TS[:, [
                i for i, col in enumerate(column_order)
                if not any(removed_col in col for removed_col in columns_to_remove)
            ]] for val_X_TS in data_X]
        else:  # train
            new_data_X = data_X[:, [
                i for i, col in enumerate(column_order) if not any(removed_col in col for removed_col in columns_to_remove)
            ]]
        new_column_order = [col for col in column_order if not any(
            removed_col in col for removed_col in columns_to_remove)]
        return new_data_X, new_column_order

    # TODO: rework or remove
    def feature_importance(self, config: str) -> None:
        '''Get feature importance values by leaving out the specified features and calculating the change in the performance'''
        feature_importance = {}
        model = MINIROCKET.MiniRocketVotingClassifier(
            n_estimators=10, n_jobs=self.n_jobs, max_dilations_per_kernel=32, class_weight=None)
        feature_search = [["REMOVE_NOTHING"], ["opensmile"],
                          ["speaker"], ["openpose"], ["openface"], ["openpose", "speaker"], ["speaker", "openpose", "openface"], ["opensmile", "speaker", "openpose"], ["opensmile", "openpose", "openface"], ["opensmile", "speaker", "openface"]]
        # per run, remove one or more columns
        for removed_cols in feature_search:
            interval_length = 900
            stride_eval = 500
            val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order, train_Y_TS_task = self.data.get_timeseries_format(
                interval_length=interval_length, stride_train=400, stride_eval=stride_eval, verbose=False, fps=50, label_creation="stride_eval")
            # remove columns ending, columns are the 2nd dimension
            train_X_TS, _ = self.remove_columns(
                columns_to_remove=removed_cols, data_X=train_X_TS, column_order=column_order)
            print(train_X_TS.shape)
            model.fit(train_X_TS, train_Y_TS[:, self.task])
            val_X_TS_list_new, _ = self.remove_columns(
                columns_to_remove=removed_cols, data_X=val_X_TS_list, column_order=column_order)
            # eval
            test_preds = self.get_full_test_preds(
                model, val_X_TS_list_new, interval_length, stride_eval)
            eval_scores = self.get_eval_metrics(
                preds=test_preds, dataset="val", verbose=False)
            name = " ".join(removed_cols)
            feature_importance[name] = eval_scores

        for key, value in feature_importance.items():
            print("Removed:", key)
            print(value["accuracy"], value["f1"], "\n")

    def learning_curve(self, config: str, iterations_per_samplesize: int, stepsize: int, save_to: str) -> None:
        '''Get learning curve of model.
        :param config: The configuration file to use to create the model.
        :param iterations_per_samplesize: Number of iterations per sample size to create an average score.
        :param stepsize: Step size for the sample sizes used for learning curve.
        :param save_to: The path to save the learning curve plot to.
        '''
        print("Learning curve run started with stepsize", stepsize, "and",
              iterations_per_samplesize, "iterations per sample size.")
        scores = []

        # load config and model
        self.config = self.read_config(self.folder+"code/"+config)
        columns_to_remove = self.column_removal_dict[self.config["data_params"]
                                                     ["columns_to_remove"]]
        model = self.get_classic_learner(self.config["model_params"])

        max_sessions = 55  # number of training files
        # start with all training data and remove one session per iteration
        for i in range(max_sessions, 0, -stepsize):
            scores_iter = []
            # generate session strings, e.g. "train_0, train_1, ..."
            sessions_train = [f"{j}_train" for j in range(i)]
            self.data.limit_to_sessions(sessions_train=sessions_train)
            print("Training on", i, "sessions...")
            for j in range(iterations_per_samplesize):
                # dataprep
                val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order, train_Y_TS_task = self.data_from_config(
                    self.config["data_params"], "timeseries", columns_to_remove, fold=4)
                print("Train X shape", train_X_TS.shape)
                print("Val X shape", val_X_TS_list[0].shape)
                # train
                model.fit(train_X_TS, train_Y_TS_task)
                # eval
                test_preds = self.get_full_test_preds(
                    model, val_X_TS_list, interval_length=self.config["data_params"]["interval_length"], stride_eval=self.config["data_params"]["stride_eval"], model_type="Classic", start_padding=self.config["data_params"]["start_padding"])
                eval_scores = self.get_eval_metrics(
                    preds=test_preds, dataset="val", verbose=False)
                scores_iter.append(eval_scores["accuracy"])
            scores.append(scores_iter)
            print("\t", scores)
        # averaged scores
        scores = np.array(scores)
        # revert order of scores
        scores = scores[::-1]
        scores_mean = np.mean(scores, axis=1)
        print("\n\nMean Scores:", scores_mean)

        # plot learning curve with standard deviation
        start_step = max_sessions % stepsize
        plt.plot(range(start_step, max_sessions, stepsize), scores_mean)
        plt.fill_between(range(0, max_sessions, stepsize), scores_mean -
                         np.std(scores, axis=1), scores_mean + np.std(scores, axis=1), alpha=0.2)
        plt.xlabel("Number of sessions in training data")
        plt.ylabel("Accuracy")
        plt.title("Learning curve")
        plt.grid(alpha=0.2)
        # save as png in plots folder
        plt.savefig(save_to)

    def get_model_values(self, trial: optuna.Trial) -> tuple:
        '''Get the model values for the trial based on the configuration and the trial parameters.
        params: trial: optuna.Trial: The trial object.
        output: model_values: dict: The model values to use for the model creation.
        output: training_values: dict: The training values to use for the model training.
        '''
        model_params = self.config["model_params"]
        model_values = {}
        training_values = {}
        if self.config["model_type"] == "RandomForest":
            model_values["n_estimators"] = trial.suggest_int(
                "n_estimators", **model_params["n_estimators"])
            model_values["max_depth"] = trial.suggest_int(
                "max_depth", **model_params["max_depth"])
            model_values["random_state"] = trial.suggest_int(
                "random_state", **model_params["random_state"])
            model_values["criterion"] = trial.suggest_categorical(
                "criterion", model_params["criterion"])
            model_values["max_features"] = trial.suggest_categorical(
                "max_features", model_params["max_features"])
        if self.config["model_type"] == "XGBoost":
            model_values["n_estimators"] = trial.suggest_int(
                "n_estimators", **model_params["n_estimators"])
            model_values["max_depth"] = trial.suggest_int(
                "max_depth", **model_params["max_depth"])
            model_values["learning_rate"] = trial.suggest_float(
                "learning_rate", **model_params["learning_rate"])
            model_values["booster"] = trial.suggest_categorical(
                "booster", model_params["booster"])
        if self.config["model_type"] == "MiniRocket":
            model_values["n_estimators"] = trial.suggest_int(
                "n_estimators", **model_params["n_estimators"])
            model_values["max_dilations_per_kernel"] = trial.suggest_int(
                "max_dilations_per_kernel", **model_params["max_dilations_per_kernel"])
            model_values["class_weight"] = trial.suggest_categorical(
                "class_weight", model_params["class_weight"])
            model_values["random_state"] = trial.suggest_categorical(
                "random_state", model_params["random_state"])
        if self.config["model_type"] == "TST":
            model_values["dropout"] = trial.suggest_float(
                "dropout", **model_params["dropout"])
            model_values["fc_dropout"] = trial.suggest_float(
                "fc_dropout", **model_params["fc_dropout"])
            model_values["n_layers"] = trial.suggest_int(
                "n_layers", **model_params["n_layers"])
            model_values["n_heads"] = trial.suggest_int(
                "n_heads", **model_params["n_heads"])
            model_values["d_model"] = trial.suggest_int(
                "d_model", **model_params["d_model"])
        if self.config["model_type"] == "LSTM_FCN":
            model_values["fc_dropout"] = trial.suggest_float(
                "fc_dropout", **model_params["fc_dropout"])
            model_values["rnn_dropout"] = trial.suggest_float(
                "rnn_dropout", **model_params["rnn_dropout"])
            model_values["hidden_size"] = trial.suggest_int(
                "hidden_size", **model_params["hidden_size"])
            model_values["rnn_layers"] = trial.suggest_int(
                "rnn_layers", **model_params["rnn_layers"])
            model_values["bidirectional"] = trial.suggest_categorical(
                "bidirectional", model_params["bidirectional"])
        if self.config["model_type"] == "ConvTranPlus":
            model_values["d_model"] = trial.suggest_int(
                "d_model", **model_params["d_model"])
            model_values["n_heads"] = trial.suggest_int(
                "n_heads", **model_params["n_heads"])
            model_values["dim_ff"] = trial.suggest_int(
                "dim_ff", **model_params["dim_ff"])
            model_values["encoder_dropout"] = trial.suggest_float(
                "encoder_dropout", **model_params["encoder_dropout"])
            model_values["fc_dropout"] = trial.suggest_float(
                "fc_dropout", **model_params["fc_dropout"])
        if self.config["model_type"] == "TransformerLSTMPlus":
            model_values["d_model"] = trial.suggest_int(
                "d_model", **model_params["d_model"])
            model_values["nhead"] = trial.suggest_int(
                "nhead", **model_params["nhead"])
            model_values["proj_dropout"] = trial.suggest_float(
                "proj_dropout", **model_params["proj_dropout"])
            model_values["num_encoder_layers"] = trial.suggest_int(
                "num_encoder_layers", **model_params["num_encoder_layers"])
            model_values["dim_feedforward"] = trial.suggest_int(
                "dim_feedforward", **model_params["dim_feedforward"])
            model_values["dropout"] = trial.suggest_float(
                "dropout", **model_params["dropout"])
            model_values["num_rnn_layers"] = trial.suggest_int(
                "num_rnn_layers", **model_params["num_rnn_layers"])
        if self.config["model_type"] in ["TST", "LSTM_FCN", "ConvTranPlus", "TransformerLSTMPlus"]:
            training_values["bs"] = trial.suggest_int(
                "bs", **model_params["bs"])
            training_values["lr"] = trial.suggest_float(
                "lr", **model_params["lr"], log=True)
            training_values["loss_func"] = trial.suggest_categorical(
                "loss", model_params["loss"])
        return model_values, training_values

    def get_tsai_learner(self, dls: object, model_values: dict, training_values: dict) -> object:
        """Get a tsai learner based on the configuration and the trial parameters.
        params: dls: object: The dataloaders object passed to the model and learner.
        params: model_values: dict: The model values to use for the model creation.
        params: training_values: dict: The training values to use for the model training.
        output: object: The tsai learner object to use for training.
        """
        model_trial_params = {}
        if self.config["model_type"] == "TST":
            model = TST(dls.vars, dls.c, dls.len, **model_values)
        elif self.config["model_type"] == "LSTM_FCN":
            model = LSTM_FCN(dls.vars, dls.c, dls.len,
                             **model_values)
        elif self.config["model_type"] == "ConvTranPlus":
            model = ConvTranPlus(dls.vars, dls.c, dls.len,
                                 **model_values)
        elif self.config["model_type"] == "TransformerLSTMPlus":
            model = TransformerLSTMPlus(dls.vars, dls.c, dls.len,
                                        **model_trial_params)

        loss_func = self.loss_dict[training_values["loss_func"]]
        cbs = [EarlyStoppingCallback(monitor="accuracy", patience=3)]
        learn = Learner(dls, model, metrics=[
            accuracy, F1Score()], cbs=cbs, loss_func=loss_func)
        return learn

    def get_classic_learner(self, model_values: dict) -> object:
        '''Get a classic learner following sklearn conventions based on the configuration and the trial parameters.
        params: model_values: dict: The model values to use for the model creation.
        output: object: The classic learner object to use for training.
        '''
        if self.config["model_type"] == "RandomForest":
            model = RandomForestClassifier(**model_values, n_jobs=self.n_jobs)
        elif self.config["model_type"] == "XGBoost":
            model = XGBClassifier(**model_values)
        elif self.config["model_type"] == "MiniRocket":
            model = MINIROCKET.MiniRocketVotingClassifier(
                **model_values, n_jobs=self.n_jobs)
        return model

    def optuna_objective_tsai(self, trial: optuna.Trial) -> tuple:
        '''Optuna objective function for all tsai style models. Optimizes for accuracy and macro f1 score.
        params: trial: optuna.Trial: The optuna trial runnning.
        output: tuple: Tuple containing the accuracy and macro f1 score of that trial run.
        '''
        accuracies = []
        f1s = []
        data_values, columns_to_remove = self.get_data_values(trial)
        model_values, training_values = self.get_model_values(trial)
        ### DATA PRE-PROCESSING ###
        for fold in range(1, 5):
            print("\nFold", fold)
            val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order, train_Y_TS_task = self.data_from_config(
                data_values=data_values, format="timeseries", columns_to_remove=columns_to_remove, fold=fold)
            all_X, all_Y, splits = self.merge_val_train(
                val_X_TS_list=val_X_TS_list, val_Y_TS_list=val_Y_TS_list, train_X_TS=train_X_TS, train_Y_TS_task=train_Y_TS_task)
            tfms = [None, TSClassification()]
            dsets = TSDatasets(all_X, all_Y, splits=splits,
                               inplace=False, tfms=tfms)
            batch_tfms = [TSStandardize(by_sample=True)]
            dls = TSDataLoaders.from_dsets(
                dsets.train, dsets.valid, bs=training_values["bs"], batch_tfms=batch_tfms)

            ### MODEL SPECIFICATION ###
            torch.cuda.empty_cache()
            learn = self.get_tsai_learner(
                dls=dls, model_values=model_values, training_values=training_values)

            learn.fit_one_cycle(100, training_values["lr"])

            ### EVALUATION ###
            preds = self.get_full_test_preds(model=learn, val_X_TS_list=val_X_TS_list, interval_length=data_values[
                "interval_length"], stride_eval=data_values["stride_eval"], model_type="TSAI", batch_tfms=batch_tfms, start_padding=data_values["start_padding"])
            outcomes = self.get_eval_metrics(
                preds=preds, dataset="val", verbose=False)

            accuracies.append(outcomes["accuracy"])
            f1s.append(outcomes["f1"])

        return np.mean(accuracies), np.mean(f1s)

    def optuna_objective_classic(self, trial: optuna.Trial) -> tuple:
        '''Optuna objective function for all classic (sklearn API style) models. Optimizes for accuracy and macro f1 score.
        params: trial: optuna.Trial: The optuna trial runnning.
        output: tuple: Tuple containing the accuracy and macro f1 score of that trial run.
        '''
        accuracies = []
        f1s = []
        # same data and model values for all folds (only choose hyperparamters once)
        data_values, columns_to_remove = self.get_data_values(trial)
        model_values, _ = self.get_model_values(trial)
        print("Model Values:", model_values)
        print("\nData Values:", data_values)
        # cross validation
        for fold in range(1, 5):
            print("\nFold", fold)
            if self.config["model_type"] == "MiniRocket":
                val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order, train_Y_TS_task = self.data_from_config(
                    data_values=data_values, format="timeseries", columns_to_remove=columns_to_remove, fold=fold)
            else:
                val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order, train_Y_TS_task = self.data_from_config(
                    data_values=data_values, format="classic", columns_to_remove=columns_to_remove, fold=fold)

            model = self.get_classic_learner(model_values)

            model.fit(train_X_TS, train_Y_TS_task)

            test_preds = self.get_full_test_preds(
                model, val_X_TS_list, data_values["interval_length"], data_values["stride_eval"], model_type="Classic", start_padding=data_values["start_padding"])

            eval_scores = self.get_eval_metrics(
                test_preds, dataset="val", verbose=True)

            accuracies.append(eval_scores["accuracy"])
            f1s.append(eval_scores["f1"])

        print("Accuracies:", accuracies)

        return np.mean(accuracies), np.mean(f1s)

    def load_and_eval(self, config_name: str, saved_model_name: str = "") -> None:
        """
        Load a model from disk and evaluate it on the test data.
        params: model_name: str: The name of the model to load.
        params: save_name: str: The file name to save the test predictions to.
        """
        if saved_model_name == "":
            saved_model_name = config_name

        # the model
        if "MiniRocket" in config_name:
            prefix = "code/trained_models/best_"
        else:
            prefix = "code/trained_models/"
        with open(self.folder + prefix + saved_model_name + ".pkl", "rb") as f:
            model = pickle.load(f)

        # features the model was trained on
        with open(self.folder + prefix + saved_model_name + "_columns.pkl", "rb") as f:
            features = pickle.load(f)

        # load config to get data preprocessing parameters
        # with open(self.folder + "code/best_model_configs/" + model_name + ".json", "r") as f:
        #    config = json.load(f)
        config = self.read_config(
            self.folder+"code/best_model_configs/"+config_name+".json")  # sets task automatically
        print("Task active:", self.task)

        data_values = config["data_params"]
        interval_length = data_values["interval_length"]
        stride_eval = data_values["stride_eval"]
        stride_train = data_values["stride_train"]
        fps = data_values["fps"]
        label_creation = data_values["label_creation"]
        task = config["task"]
        rescaling = data_values["rescaling"]
        columns_to_remove = data_values["columns_to_remove"]
        columns_to_remove = self.column_removal_dict[columns_to_remove]

        val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order = self.data.get_timeseries_format(interval_length=interval_length, stride_train=stride_train,
                                                                                                             stride_eval=stride_eval, fps=fps, verbose=True, label_creation=label_creation, task=task, rescaling=rescaling, fold=4)

        print(train_Y_TS.shape)  # Debugging
        print(train_Y_TS[0])
        test_X_TS_list, _ = self.data.get_timeseries_format_test_data(interval_length=interval_length, stride_eval=stride_eval,
                                                                      fps=fps, verbose=True, label_creation=label_creation, task=task, rescaling=rescaling)
        print(len(features))
        print(test_X_TS_list[0].shape)

        # nan handling based on training parameters
        if data_values["nan_handling"] == "zeros":
            train_X_TS = np.nan_to_num(train_X_TS, nan=0)
            val_X_TS_list = [np.nan_to_num(val_X_TS, nan=0)
                             for val_X_TS in val_X_TS_list]
            test_X_TS_list = [np.nan_to_num(test_X_TS, nan=0)
                              for test_X_TS in test_X_TS_list]
        if data_values["nan_handling"] == "avg":
            train_X_TS = DataLoader_HRI.impute_nan_with_feature_mean(
                train_X_TS)
            val_X_TS_list = [DataLoader_HRI.impute_nan_with_feature_mean(
                val_X_TS) for val_X_TS in val_X_TS_list]
            test_X_TS_list = [DataLoader_HRI.impute_nan_with_feature_mean(
                test_X_TS) for test_X_TS in test_X_TS_list]

        # feature removal based on training parameters
        test_X_TS_list, new_column_order = self.remove_columns(columns_to_remove=columns_to_remove,
                                                               data_X=test_X_TS_list, column_order=column_order)
        val_X_TS_list, new_column_order = self.remove_columns(columns_to_remove=columns_to_remove,
                                                              data_X=val_X_TS_list, column_order=column_order)
        print("Feature shape after removal")
        print(len(features))
        print(test_X_TS_list[0].shape)

        test_preds = self.get_full_test_preds(
            model, test_X_TS_list, interval_length=interval_length, stride_eval=stride_eval, model_type="Classic", start_padding=data_values["start_padding"])
        val_preds = self.get_full_test_preds(
            model, val_X_TS_list, interval_length=interval_length, stride_eval=stride_eval, model_type="Classic", start_padding=data_values["start_padding"])
        val_scores = self.get_eval_metrics(
            val_preds, dataset="val", verbose=True)
        print("Val Scores:", val_scores)
        test_preds_df = pd.DataFrame(test_preds).T
        test_preds_df.to_csv(self.folder + "code/test_predictions/" +
                             config_name + "_test_preds_trainer.csv")

    def train_and_save_best_model(self, model_config: str, name_extension="", fold: int = 4) -> None:
        """Train a model based on the specified configuration and save it to disk. For final submission.
        params: model_config: str: The name of the model configuration file to use for training.
        """

        self.config = self.read_config(
            self.folder+"code/best_model_configs/"+model_config)
        print(self.config)

        columns_to_remove = self.column_removal_dict[self.config["data_params"]
                                                     ["columns_to_remove"]]

        if any(s in self.config["model_type"] for s in ["TST", "LSTM_FCN", "ConvTranPlus", "TransformerLSTMPlus"]):
            return NotImplementedError

        elif any(s in self.config["model_type"] for s in ["RandomForest", "XGBoost", "MiniRocket"]):
            model = self.get_classic_learner(self.config["model_params"])
            format = "timeseries" if "MiniRocket" in self.config["model_type"] else "classic"
            val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order, train_Y_TS_task = self.data_from_config(
                self.config["data_params"], format=format, columns_to_remove=columns_to_remove, fold=fold)

            print("Column order used:", column_order)
            model.fit(train_X_TS, train_Y_TS_task)
            if fold in [1, 2, 3, 4]:  # only validate if not trained on all data
                test_preds = self.get_full_test_preds(
                    model, val_X_TS_list, self.config["data_params"]["interval_length"], self.config["data_params"]["stride_eval"], model_type="Classic", start_padding=self.config["data_params"]["start_padding"])
                eval_scores = self.get_eval_metrics(
                    test_preds, dataset="val", verbose=True)
                print("Val Scores:", eval_scores)

            # save model and column order
            if str(self.config["model_type"]) not in name_extension:
                name_extension = str(
                    self.config["model_type"]) + name_extension
            with open(self.folder+"code/trained_models/"+name_extension+".pkl", "wb") as f:
                pickle.dump(model, f)
            with open(self.folder+"code/trained_models/"+name_extension+"_columns.pkl", "wb") as f:
                pickle.dump(column_order, f)

        else:
            raise Exception("Model type not recognized.")


if __name__ == '__main__':
    my_setup(optuna)
    print(platform.platform())
    # parse arguments (config file, n_jobs)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.",
                        default="search_configs/config_minirocket.json")
    parser.add_argument(
        "--njobs", type=int, help="Number of cpu cores to use for training.", default=8)
    args = parser.parse_args()
    print(args)
    if args.config:
        config_name = args.config
    if args.njobs:
        n_jobs = args.njobs

    print("n_jobs:", n_jobs, "\nconfig:", config_name)

    if os.getcwd().endswith("HRI-Error-Detection-STAI"):
        pathprefix = ""
    else:
        pathprefix = "HRI-Error-Detection-STAI/"

    trainer = TS_Model_Trainer(
        folder=pathprefix, n_jobs=n_jobs, config_name=config_name)

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H")

    ########### uncomment to run a single training ###########
    # trainer.train_and_save_best_model(
    #    "MiniRocket_2024-06-19-08.json", name_extension="trained_on_all_data", fold=5)

    ######### get test predictions for best model ###########
    # TASK 0
    # trainer.load_and_eval("MiniRocket_UA-Submission")
    # TASK 1
    # trainer.load_and_eval("MiniRocket_RM-Submission")
    # TASK 2
    # newer: "MiniRocket_2024-06-19-08"
    # trainer.load_and_eval("MiniRocket_IR-Submission1")
    # trainer.load_and_eval("MiniRocket_IR-Submission2")
    # trainer.load_and_eval("RandomForest_2024-06-05-17")

    ########### uncomment to run optuna search ###########
    study_name = trainer.config["model_type"] + "_" + date
    # study = trainer.optuna_study(
    #    n_trials=trainer.config["n_trials"], model_type=trainer.config["model_type"], study_name=study_name, verbose=True)

    # TODO: move to analysis script
    # feature importance
    # trainer.feature_importance()

    # learning curve
    trainer.learning_curve("best_model_configs/MiniRocket_2024-06-25-17.json",
                           iterations_per_samplesize=8, stepsize=3, save_to=pathprefix+"plots/learning_curve.pdf")
