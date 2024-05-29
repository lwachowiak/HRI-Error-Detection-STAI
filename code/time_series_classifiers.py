import optuna.study.study
from tsai.models import MINIROCKET, HydraMultiRocketPlus, TST, MINIROCKET_Pytorch
from tsai.learner import ts_learner
from data_loader import DataLoader_HRI
from tsai.data.all import *
from tsai.models.utils import *
from tsai.all import my_setup, ShowGraphCallback2, LabelSmoothingCrossEntropyFlat, RocAucBinary, accuracy, F1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import optuna
from optuna.integration import WeightsAndBiasesCallback
import wandb
import json
import datetime
import platform
import numpy as np
from get_metrics import get_metrics
import matplotlib.pyplot as plt

# TODO:
# - just train with some columns / column selection / feature importance
# Models:
#   - try different models: TST, HydraMultiRocketPlus, RandomForest, XGBoost
#   - Try annotation/outlier processing: https://www.sktime.net/en/stable/api_reference/annotation.html
#   - Try prediction/forcasting --> unlikely sequence --> outlier --> label 1


class TS_Model_Trainer:

    def __init__(self, data_folder: str, task: int, n_jobs: int):
        '''
        Initialize the TS_Model_Trainer with the data folder and the task to be trained on.
        :param data_folder: The folder containing the data.
        :param task: The task to be trained on. 0: UserAwkwardness, 1: RobotMistake, 2: InteractionRupture
        :param n_jobs: CPU usage parameter
        '''
        self.data = DataLoader_HRI(data_folder)
        self.task = task
        self.TASK_TO_COLUMN = {0: "UserAwkwardness",
                               1: "RobotMistake", 2: "InteractionRupture"}
        self.n_jobs = n_jobs
        self.objective_per_model = {
            "MiniRocket": self.optuna_objective_minirocket,
            "MiniRocketTorch": self.optuna_objective_minirocketTorch,
            # "TST": self.optuna_objective_tst
        }
        self.config = None
        self.column_removal_dict = {"REMOVE_NOTHING": ["REMOVE_NOTHING"],
                                    "opensmile": ["opensmile"],
                                    "speaker": ["speaker"],
                                    "openpose": ["openpose"],
                                    "openface": ["openface"],
                                    "openpose, speaker": ["openpose", "speaker"],
                                    "speaker, openpose, openface": ["speaker", "openpose", "openface"]
                                    }

    def get_full_test_preds(self, model: object, val_X_TS_list: list, intervallength: int, stride_eval: int) -> list:
        '''Get full test predictions by repeating the predictions based on intervallength and stride_eval.
        :param model: The model to evaluate.
        :param val_X_TS_list: List of validation/test data per session.
        :param intervallength: The length of the interval to predict.
        :param stride_eval: The stride to evaluate the model.
        '''
        test_preds = []
        for val_X_TS in val_X_TS_list:  # per session
            pred = model.predict(val_X_TS)
            # for each sample in the session, repeat the prediction based on intervallength and stride_eval
            processed_preds = []
            for i, pr in enumerate(pred):
                if i == 0:
                    # first prediction, so append it intervallength times
                    processed_preds.extend([pr]*intervallength)
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
        for i in range(len(preds)):
            session_id = i
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
        print(eval_scores)

        # print confusion matrix
        if verbose:
            print(confusion_matrix(y_true, preds, normalize='all'))
            print(eval_scores)

        return eval_scores

    def data_from_config(self, config: dict, trial: optuna.Trial):
        """
        create the datasets for training based on the configuration and the trial parameters.
        params: config: dict: The configuration dictionary.
        params: trial: optuna.Trial: The trial object.
        output: tuple: Tuple containing the validation and training datasets.
        """
        data_params = self.config["data_params"]
        data_values = {}
        data_values["intervallength"] = trial.suggest_int(
            "intervallength", low=data_params["intervallength"]["low"], high=data_params["intervallength"]["high"], step=data_params["intervallength"]["step"])
        # strides must be leq than intervallength
        data_values["stride_train"] = trial.suggest_int(
            "stride_train", low=data_params["stride_train"]["low"], high=min(data_values["intervallength"], data_params["stride_train"]["high"]), step=data_params["intervallength"]["step"])
        data_values["stride_eval"] = trial.suggest_int(
            "stride_eval", low=data_params["stride_eval"]["low"], high=min(data_values["intervallength"], data_params["stride_eval"]["high"]), step=data_params["intervallength"]["step"])
        data_values["fps"] = trial.suggest_categorical(
            "fps", data_params["fps"])
        data_values["columns_to_remove"] = trial.suggest_categorical("columns_to_remove",
                                                                     data_params["columns_to_remove"])
        columns_to_remove = self.column_removal_dict[data_values["columns_to_remove"]]
        data_values["label_creation"] = trial.suggest_categorical(
            "label_creation", data_params["label_creation"])
        data_values["nan_handling"] = trial.suggest_categorical(
            "nan_handling", data_params["nan_handling"])

        # get timeseries format
        val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order = self.data.get_timeseries_format(
            intervallength=data_values["intervallength"], stride_train=data_values["stride_train"], stride_eval=data_values["stride_eval"], verbose=False, fps=data_values["fps"], label_creation=data_values["label_creation"])
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
        train_X_TS = self.remove_columns(columns_to_remove=columns_to_remove,
                                         data_X=train_X_TS, column_order=column_order)
        val_X_TS_list = self.remove_columns(columns_to_remove=columns_to_remove,
                                            data_X=val_X_TS_list, column_order=column_order)

        train_Y_TS_task = train_Y_TS[:, self.task]

        return val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order, train_Y_TS_task, data_values

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
            val_Y_TS = val_Y_TS[:, 2]
            all_Y = np.concatenate((all_Y, val_Y_TS), axis=0)
        # print(all_X.shape, all_Y.shape)
        splits = [range(0, len(train_X_TS)), range(
            len(train_X_TS), len(all_X))]
        return all_X, all_Y, splits

    def optuna_objective_minirocketTorch(self, trial: optuna.Trial) -> tuple:
        model_params = self.config["model_params"]
        # model params
        bs = trial.suggest_int("bs", low=model_params["batch_size"]["low"],
                               high=model_params["batch_size"]["high"], step=model_params["batch_size"]["step"])
        lr = trial.suggest_float("lr", low=1e-5, high=1e-1, log=True)

        val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order, train_Y_TS_task, data_values = self.data_from_config(
            self.config, trial)

        all_X, all_Y, splits = self.merge_val_train(
            val_X_TS_list=val_X_TS_list, val_Y_TS_list=val_Y_TS_list, train_X_TS=train_X_TS, train_Y_TS_task=train_Y_TS_task)
        tfms = [None, TSClassification()]
        batch_tfms = TSStandardize(by_sample=True)
        dsets = TSDatasets(all_X, all_Y, splits=splits, inplace=True)
        dls = TSDataLoaders.from_dsets(
            dsets.train, dsets.valid, batch_tfms=batch_tfms, bs=8)
        model = build_ts_model(MINIROCKET_Pytorch.MiniRocket, dls=dls)
        learn = ts_learner(dls, model, metrics=[accuracy, F1Score()], cbs=None)
        learn.fit_one_cycle(10, lr)
        acc, f1 = learn.recorder.values[-1][-2], learn.recorder.values[-1][-1]
        return acc, f1

    def optuna_objective_minirocket(self, trial: optuna.Trial) -> tuple:
        '''Optuna objective function for MiniRocket model. Optimizes for accuracy and macro f1 score.
        params: trial: optuna.Trial: The optuna trial runnning.
        output: tuple: Tuple containing the accuracy and macro f1 score of that trial run.
        '''
        model_params = self.config["model_params"]
        # # data params
        # data_params = self.config["data_params"]
        # intervallength = trial.suggest_int(
        #     "intervallength", low=data_params["intervallength"]["low"], high=data_params["intervallength"]["high"], step=data_params["intervallength"]["step"])
        # # stride must be leq than intervallength
        # stride_train = trial.suggest_int(
        #     "stride_train", low=data_params["stride_train"]["low"], high=min(intervallength, data_params["stride_train"]["high"]), step=data_params["intervallength"]["step"])
        # stride_eval = trial.suggest_int(
        #     "stride_eval", low=data_params["stride_eval"]["low"], high=min(intervallength, data_params["stride_eval"]["high"]), step=data_params["intervallength"]["step"])
        # fps = trial.suggest_categorical("fps", data_params["fps"])
        # columns_to_remove = trial.suggest_categorical("columns_to_remove",
        #                                               data_params["columns_to_remove"])
        # columns_to_remove = self.column_removal_dict[columns_to_remove]
        # label_creation = trial.suggest_categorical(
        #     "label_creation", data_params["label_creation"])
        # nan_handling = trial.suggest_categorical(
        #     "nan_handling", data_params["nan_handling"])
        # # get timeseries format
        # val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order = self.data.get_timeseries_format(
        #     intervallength=intervallength, stride_train=stride_train, stride_eval=stride_eval, verbose=False, fps=fps, label_creation=label_creation)
        # # nan handling
        # if nan_handling == "zeros":
        #     train_X_TS = np.nan_to_num(train_X_TS, nan=0)
        #     val_X_TS_list = [np.nan_to_num(val_X_TS, nan=0)
        #                      for val_X_TS in val_X_TS_list]
        # if nan_handling == "avg":
        #     train_X_TS = DataLoader_HRI.impute_nan_with_feature_mean(
        #         train_X_TS)
        #     val_X_TS_list = [DataLoader_HRI.impute_nan_with_feature_mean(
        #         val_X_TS) for val_X_TS in val_X_TS_list]
        # # feature removal
        # train_X_TS = self.remove_columns(columns_to_remove=columns_to_remove,
        #                                  data_X=train_X_TS, column_order=column_order)
        # val_X_TS_list = self.remove_columns(columns_to_remove=columns_to_remove,
        #                                     data_X=val_X_TS_list, column_order=column_order)

        # train_Y_TS_task = train_Y_TS[:, self.task]

        val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order, train_Y_TS_task, data_values = self.data_from_config(
            self.config, trial)

        # model params
        max_dilations_per_kernel = trial.suggest_int(
            "max_dilations_per_kernel", low=model_params["max_dilations_per_kernel"]["low"], high=model_params["max_dilations_per_kernel"]["high"], step=model_params["max_dilations_per_kernel"]["step"])
        n_estimators = trial.suggest_int(
            "n_estimators", low=model_params["n_estimators"]["low"], high=model_params["n_estimators"]["high"], step=model_params["n_estimators"]["step"])
        class_weight = trial.suggest_categorical(
            "class_weight", [None])  # ["balanced", None])

        model = MINIROCKET.MiniRocketVotingClassifier(
            n_estimators=n_estimators, n_jobs=self.n_jobs, max_dilations_per_kernel=max_dilations_per_kernel, class_weight=class_weight)
        model.fit(train_X_TS, train_Y_TS_task)
        test_preds = self.get_full_test_preds(
            model, val_X_TS_list, data_values["intervallength"], data_values["stride_eval"])
        # test_preds = []
        # for val_X_TS in val_X_TS_list:  # per session
        #    pred = model.predict(val_X_TS)
        # for each sample in the session, repeat the prediction based on intervallength and stride_eval
       #     processed_preds = []
       #     for i, pr in enumerate(pred):
       #         if i == 0:
        # first prediction, so append it intervallength times
       #             processed_preds.extend([pr]*intervallength)
       #         else:
        # all other predictions are appended stride_eval times
       #             processed_preds.extend([pr]*stride_eval)
       #     test_preds.append(processed_preds)

        eval_scores = self.get_eval_metrics(
            preds=test_preds, dataset="val", verbose=True)
        return eval_scores["accuracy"], eval_scores["f1"]

    def optuna_study(self, n_trials: int, model_type: str, study_name: str, verbose=False) -> optuna.study.Study:
        """Performs an Optuna study to optimize the hyperparameters of the model.
        :param n_trials: The number of search trials to perform.
        :param model_type: The type of model to optimize (MiniRocket, TST).
        """
        wandb_kwargs = {"project": "HRI-Errors"}
        wandbc = WeightsAndBiasesCallback(
            metric_name=["accuracy", "macro f1"], wandb_kwargs=wandb_kwargs)

        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        study = optuna.create_study(
            directions=["maximize", "maximize"], study_name=date + "_" + study_name)
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
        self.get_trials_figures(study, 1, "macro f1")

        f = "best_{}".format
        for param_name, param_value in trial_with_highest_accuracy.params.items():
            wandb.run.summary[f(param_name)] = param_value

        wandb.run.summary["best accuracy"] = trial_with_highest_accuracy.values[0]
        wandb.run.summary["best macro f1"] = trial_with_highest_accuracy.values[1]

        wandb.finish()

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
        fig = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[target_index], target_name=target_name)
        wandb.log({"optuna_param_importances_"+target_name: fig})
        fig = optuna.visualization.plot_slice(
            study, target=lambda t: t.values[target_index], target_name=target_name)
        wandb.log({"optuna_slice_"+target_name: fig})
        fig = optuna.visualization.plot_parallel_coordinate(
            study, target=lambda t: t.values[target_index], target_name=target_name)
        wandb.log({"optuna_parallel_coordinate_"+target_name: fig})

    def read_config(self, file_path: str) -> dict:
        """Reads a JSON configuration file and returns the configuration as a dictionary."""
        try:
            with open(file_path, 'r') as file:
                config = json.load(file)
                # print config
                print("\nConfiguration loaded successfully.")
                for key, value in config.items():
                    print(f"{key}: {value}")
                self.config = config
                return config
        except FileNotFoundError:
            print("\nError: The configuration file was not found.")
        except json.JSONDecodeError:
            print("\nError: The configuration file is not in proper JSON format.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def retrain_best_trial(self, study: optuna.study.Study, model_type: str) -> None:
        '''Retrain the best trial of a study.'''
        best_trial = max(study.best_trials, key=lambda t: t.values[0])
        print("Best trial:")
        print("Values: ", best_trial.values, "Params: ", best_trial.params)
        if model_type == "MiniRocket":
            self.optuna_objective_minirocket(best_trial)
        elif model_type == "TST":
            self.optuna_objective_tst(best_trial)
        else:
            print("Model type not supported.")

    def remove_columns(self, columns_to_remove: list, data_X: np.array, column_order: list) -> np.array:
        '''Remove columns from the data.
        :param columns_to_remove: List of columns to remove.
        :param data_X: The data to remove the columns from. Either a list of np.arrays or a np.array.
        :param column_order: The order of the columns in the data.
        :output new_data_X: The data with the specified columns removed.
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
        return new_data_X

    def feature_importance(self):
        '''Get feature importance values by leaving out the specified features and calculating the change in the performance'''
        feature_importance = {}
        model = MINIROCKET.MiniRocketVotingClassifier(
            n_estimators=10, n_jobs=self.n_jobs, max_dilations_per_kernel=32, class_weight=None)
        feature_search = [["REMOVE_NOTHING"], ["opensmile"],
                          ["speaker"], ["openpose"], ["openface"], ["openpose", "speaker"], ["speaker", "openpose", "openface"], ["opensmile", "speaker", "openpose"], ["opensmile", "openpose", "openface"], ["opensmile", "speaker", "openface"]]
        # per run, remove one or more columns
        for removed_cols in feature_search:
            intervallength = 900
            stride_eval = 500
            val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order = self.data.get_timeseries_format(
                intervallength=intervallength, stride_train=400, stride_eval=stride_eval, verbose=False, fps=50, label_creation="stride_eval")
            # remove columns ending, columns are the 2nd dimension
            train_X_TS = self.remove_columns(
                columns_to_remove=removed_cols, data_X=train_X_TS, column_order=column_order)
            print(train_X_TS.shape)
            model.fit(train_X_TS, train_Y_TS[:, self.task])
            val_X_TS_list_new = self.remove_columns(
                columns_to_remove=removed_cols, data_X=val_X_TS_list, column_order=column_order)
            # eval
            test_preds = self.get_full_test_preds(
                model, val_X_TS_list_new, intervallength, stride_eval)
            eval_scores = self.get_eval_metrics(
                preds=test_preds, dataset="val", verbose=False)
            name = " ".join(removed_cols)
            feature_importance[name] = eval_scores

        for key, value in feature_importance.items():
            print("Removed:", key)
            print(value["accuracy"], value["f1"], "\n")

    def learning_curve(self, iterations_per_samplesize: int, stepsize: int, save_to: str) -> None:
        '''Get learning curve of model.
        :param iterations_per_samplesize: Number of iterations per sample size to create an average score.
        :param stepsize: Step size for the sample sizes used for learning curve.
        '''
        print("Learning curve run started with stepsize", stepsize, "and",
              iterations_per_samplesize, "iterations per sample size.")
        scores = []
        model = MINIROCKET.MiniRocketVotingClassifier(
            n_estimators=5, n_jobs=self.n_jobs, max_dilations_per_kernel=32, class_weight=None)
        intervallength = 800
        stride_eval = 400
        stride_train = 400
        fps = 25
        label_creation = "stride_eval"
        columns_to_remove = ["openpose"]
        max_sessions = len(self.data.train_X["session"].unique())
        # start with all training data and remove one session per iteration
        for i in range(max_sessions, 0, -stepsize):
            scores_iter = []
            self.data.limit_to_sessions(sessions_train=range(0, i))
            print("Training on", i, "sessions...")
            for j in range(iterations_per_samplesize):
                # dataprep
                val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order = self.data.get_timeseries_format(
                    intervallength=intervallength, stride_train=stride_train, stride_eval=stride_eval, verbose=False, fps=fps, label_creation=label_creation)
                train_X_TS = self.remove_columns(
                    columns_to_remove=columns_to_remove, data_X=train_X_TS, column_order=column_order)
                val_X_TS_list_new = self.remove_columns(
                    columns_to_remove=columns_to_remove, data_X=val_X_TS_list, column_order=column_order)
                # train
                model.fit(train_X_TS, train_Y_TS[:, self.task])
                # eval
                test_preds = self.get_full_test_preds(
                    model, val_X_TS_list_new, intervallength, stride_eval)
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


if __name__ == '__main__':
    my_setup(optuna)
    print(platform.platform())
    if "macOS" in platform.platform():
        n_jobs = 4
        pathprefix = ""
        config_name = "configs/config_mac.json"
    else:
        n_jobs = -1
        pathprefix = "HRI-Error-Detection-STAI/"
        config_name = "configs/config_minirocket.json"
    print("n_jobs:", n_jobs)

    trainer = TS_Model_Trainer(pathprefix+"data/", task=2, n_jobs=n_jobs)
    config = trainer.read_config(pathprefix+"code/"+config_name)

    study = trainer.optuna_study(
        n_trials=config["n_trials"], model_type=config["model_type"], study_name=config["model_type"], verbose=True)

    # repeat best trial
    # trainer.retrain_best_trial(study, config["model_type"])

    # feature importance
    # trainer.feature_importance()

    # learning curve
    # trainer.learning_curve(iterations_per_samplesize=8, stepsize=3, save_to=pathprefix+"plots/learning_curve.pdf")
