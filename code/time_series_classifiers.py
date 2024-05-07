from tsai.models import MINIROCKET
from data_loader import DataLoader_HRI
from tsai.data.all import *
from tsai.models.utils import *
from tsai.all import my_setup
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import optuna
from optuna.integration import WeightsAndBiasesCallback
import wandb
import json
import datetime
import platform
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# TODO:
# - true eval on full lenght
# - just train with some columns / column selection
# - try different models
# - more minirocket params
# - https://www.sktime.net/en/stable/api_reference/annotation.html

# Models to try: HydraMultiRocketPlus


# def (data):
#   # Preprocess the data
#   return data

class TS_Model_Trainer:

    def __init__(self, data_folder, task, n_jobs):
        self.data = DataLoader_HRI(data_folder)
        self.task = task
        self.TASK_TO_COLUMN = {0: "UserAwkwardness",
                               1: "RobotMistake", 2: "InteractionRupture"}
        self.n_jobs = n_jobs
        self.objective_per_model = {
            "MiniRocket": self.optuna_objective_minirocket}
        self.config = None

    def evaluate_model(self, preds, dataset="val", verbose=False):
        '''Evaluate model on self.data.val_X and self.data.val_Y. The final missing values in preds are filled with 0s.'''
        # compare length between preds and val_Y and append 0s to preds if necessary

        # TODO: remove
       # temp_val_session_ids = [0, 3, 4, 7, 8, 11, 12, 13]
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
                    print("Appended,", to_append,
                          "0s to preds for session", session_id, "resulting in", len(preds[i]), "predictions")
            if (len(preds[i]) != len(self.data.val_Y[self.data.val_Y['session'] == session_id])):
                print("ERROR: Length of preds and val_Y do not match for session", session_id,
                      "preds:", len(preds[i]), "val_Y:", len(self.data.val_Y[self.data.val_Y['session'] == session_id]))
        # flatten preds
        preds = np.concatenate(preds)
        y_true = np.concatenate(y_true)
        print("Final preds length", len(preds), len(y_true))
        # TODO: THIS DOESNT MAKE SENSE!!!!! I NEED TO ADD 0s per SESSION, not for the whole dataset
        # if len(preds) < len(self.data.val_Y):
        #    to_append = len(self.data.val_Y)-len(preds)
        #    preds = np.append(preds, [0]*(to_append))
        #    if verbose:
        #        print("Appended,", to_append, "0s to preds")
        # get task column of val_Y_TS
        # val_Y_task = self.data.val_Y[:, self.task]
        # get eval scores

        eval_scores = {}
        eval_scores["accuracy"] = accuracy_score(y_true=y_true, y_pred=preds)
        eval_scores["macro f1"] = f1_score(
            y_true=y_true, y_pred=preds, average='macro')
        eval_scores["macro precision"] = precision_score(
            y_true=y_true, y_pred=preds, average='macro')
        eval_scores["macro recall"] = recall_score(
            y_true=y_true, y_pred=preds, average='macro')

        # print confusion matrix
        if verbose:
            print(confusion_matrix(y_true, preds))
            print(eval_scores)

        return eval_scores

    def optuna_objective_minirocket(self, trial: optuna.Trial):
        '''Optuna objective function for MiniRocket model. Optimizes for accuracy and macro f1 score.'''
        # parameters being optimized
        # data params
        params = self.config["mini_rocket_params"]
        intervallength = trial.suggest_int(
            "intervallength", low=params["intervallength"]["low"], high=params["intervallength"]["high"], step=params["intervallength"]["step"])
        # stride must be leq than intervallength
        stride = trial.suggest_int(
            "stride", low=params["stride"]["low"], high=intervallength, step=params["intervallength"]["step"])
        fps = trial.suggest_categorical("fps", params["fps"])

        # model params
        max_dilations_per_kernel = trial.suggest_int(
            "max_dilations_per_kernel", low=params["max_dilations_per_kernel"]["low"], high=params["max_dilations_per_kernel"]["high"], step=params["max_dilations_per_kernel"]["step"])
        n_estimators = trial.suggest_int(
            "n_estimators", low=params["n_estimators"]["low"], high=params["n_estimators"]["high"], step=params["n_estimators"]["step"])
        class_weight = trial.suggest_categorical(
            "class_weight", ["balanced", None])

        # get timeseries format
        val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS = self.data.get_timeseries_format(
            intervallength=intervallength, stride=stride, verbose=False, fps=fps)

        train_Y_TS_task = train_Y_TS[:, self.task]

        model = MINIROCKET.MiniRocketVotingClassifier(
            n_estimators=n_estimators, n_jobs=self.n_jobs, max_dilations_per_kernel=max_dilations_per_kernel, class_weight=class_weight)
        model.fit(train_X_TS, train_Y_TS_task)
        test_preds = []
        for val_X_TS in val_X_TS_list:
            pred = model.predict(val_X_TS)
            pred = np.repeat(pred, intervallength)
            test_preds.append(pred)
        # test_preds = model.predict(val_X_TS)
        # print(test_preds)

        # print("Original Test acc")  # TODO remove debugging
        # test_acc = model.score(val_X_TS, val_Y_TS_task)
        # print(test_acc)
        # print(confusion_matrix(val_Y_TS_task, test_preds))
        # print("New Eval")

        # expand test_preds to full length based on intervallength, i.e., each prediction is repeated intervallength times
        # test_preds = np.repeat(test_preds, intervallength)
        # print(test_preds, len(test_preds))
        eval_scores = self.evaluate_model(
            preds=test_preds, dataset="val", verbose=True)
        return eval_scores["accuracy"], eval_scores["macro f1"]

        # test_acc = model.score(val_X_TS, val_Y_TS_task)
        # print(confusion_matrix(val_Y_TS_task, test_preds))
        # # f1, precision, recall
        # f1 = f1_score(val_Y_TS_task, test_preds, average='macro')
        # precision = precision_score(
        #     val_Y_TS_task, test_preds, average='macro')
        # recall = recall_score(val_Y_TS_task, test_preds, average='macro')
        # print(
        #     f"Test accuracy: {test_acc}, F1: {f1}, Precision: {precision}, Recall: {recall}")
        # return test_acc, f1

    def optuna_study(self, n_trials, model_type, study_name, verbose=False):
        """Performs an Optuna study to optimize the hyperparameters of the model."""
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

    def get_trials_figures(self, study, target_index, target_name):
        '''Get optuna visualization figures and log them to wandb. Summary visualizations for full search.'''
        fig = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[target_index], target_name=target_name)
        wandb.log({"optuna_optimization_history_"+target_name: fig})
        fig = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[target_index], target_name=target_name)
        wandb.log({"optuna_param_importances_"+target_name: fig})
        fig = optuna.visualization.plot_slice(
            study, target=lambda t: t.values[target_index], target_name=target_name)
        wandb.log({"optuna_slice_"+target_name: fig})
        fig = optuna.visualization.plot_parallel_coordinate(
            study, target=lambda t: t.values[target_index], target_name=target_name)
        wandb.log({"optuna_parallel_coordinate_"+target_name: fig})

    def read_config(self, file_path):
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


if __name__ == '__main__':
    my_setup(optuna)
    print(platform.platform())
    if "macOS" in platform.platform():
        n_jobs = 2
        pathprefix = ""
        config_name = "config_mac.json"
    else:
        n_jobs = -1
        pathprefix = "HRI-Error-Detection-STAI/"
        config_name = "config.json"
    print("n_jobs:", n_jobs)

    trainer = TS_Model_Trainer(pathprefix+"data/", task=2, n_jobs=n_jobs)
    config = trainer.read_config(pathprefix+"code/"+config_name)

    # trainer.data.limit_to_sessions(
    #    sessions_train=[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], sessions_val=[0, 3, 4, 7, 8, 11, 12, 13])

    study = trainer.optuna_study(
        n_trials=config["n_trials"], model_type=config["model_type"], study_name=config["model_type"], verbose=True)
