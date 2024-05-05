from tsai.models import MINIROCKET
from data_loader import DataLoader_HRI
from tsai.data.all import *
from tsai.models.utils import *
from tsai.all import my_setup
import pandas as pd
from sklearn.metrics import confusion_matrix
import optuna
from optuna.integration import WeightsAndBiasesCallback
import wandb
import json
import datetime
import platform
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# TODO:
# - true eval on full lenght
# - just train with some columns / column selection
# - try different models
# - optimize for f1?
# - more minirocket params

# Models to try: HydraMultiRocketPlus


# def (data):
#   # Preprocess the data
#   return data

class TS_Model_Trainer:

    def __init__(self, data_folder, task, n_jobs):
        self.data = DataLoader_HRI(data_folder)
        self.task = task
        self.n_jobs = n_jobs
        self.objective_per_model = {
            "MiniRocket": self.optuna_objective_minirocket}

    def evaluate_model(model, data):
        # Evaluate the model
        return None

    def optuna_objective_minirocket(self, trial: optuna.Trial):
        stride = trial.suggest_int("stride", low=500, high=1000, step=50)

        # parameters being optimized
        max_dilations_per_kernel = trial.suggest_int(
            "max_dilations_per_kernel", low=4, high=256, step=8)
        val_X_TS, val_Y_TS, train_X_TS, train_Y_TS = self.data.get_timeseries_format(
            intervallength=1000, stride=stride, verbose=False)
        n_estimators = trial.suggest_int(
            "n_estimators", low=4, high=12, step=2)
        class_weight = trial.suggest_categorical(
            "class_weight", ["balanced", None])

        train_Y_TS_task = train_Y_TS[:, self.task]
        val_Y_TS_task = val_Y_TS[:, self.task]

        model = MINIROCKET.MiniRocketVotingClassifier(
            n_estimators=n_estimators, n_jobs=self.n_jobs, max_dilations_per_kernel=max_dilations_per_kernel)
        model.fit(train_X_TS, train_Y_TS_task)

        test_acc = model.score(val_X_TS, val_Y_TS_task)
        test_preds = model.predict(val_X_TS)
        print(confusion_matrix(val_Y_TS_task, test_preds))
        # f1, precision, recall
        f1 = f1_score(val_Y_TS_task, test_preds, average='macro')
        precision = precision_score(
            val_Y_TS_task, test_preds, average='macro')
        recall = recall_score(val_Y_TS_task, test_preds, average='macro')
        print(
            f"Test accuracy: {test_acc}, F1: {f1}, Precision: {precision}, Recall: {recall}")
        # trial.set_user_attr("macro f1", f1)
        # trial.set_user_attr("macro precision", precision)
        # trial.set_user_attr("macro recall", recall)
        # wandb.log({"macro f1": f1, "macro precision": precision,
        #          "macro recall": recall})

        return test_acc, f1

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
        fig = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[0])
        wandb.log({"optuna_optimization_history": fig})
        fig = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[0])
        wandb.log({"optuna_param_importances": fig})
        fig = optuna.visualization.plot_slice(
            study, target=lambda t: t.values[0])
        wandb.log({"optuna_slice": fig})
        fig = optuna.visualization.plot_parallel_coordinate(
            study, target=lambda t: t.values[0])
        wandb.log({"optuna_parallel_coordinate": fig})

        f = "best_{}".format
        for param_name, param_value in trial_with_highest_accuracy.params.items():
            wandb.run.summary[f(param_name)] = param_value

        wandb.run.summary["best accuracy"] = trial_with_highest_accuracy.values[0]
        wandb.run.summary["best macro f1"] = trial_with_highest_accuracy.values[1]

        wandb.finish()

        return study

    @ staticmethod
    def read_config(file_path):
        """Reads a JSON configuration file and returns the configuration as a dictionary."""
        try:
            with open(file_path, 'r') as file:
                config = json.load(file)
                # print config
                print("Configuration loaded successfully.")
                for key, value in config.items():
                    print(f"{key}: {value}")
                return config
        except FileNotFoundError:
            print("Error: The configuration file was not found.")
        except json.JSONDecodeError:
            print("Error: The configuration file is not in proper JSON format.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    my_setup(optuna)
    print(platform.platform())
    if "macOS" in platform.platform():
        n_jobs = 2
        pathprefix = ""
    else:
        n_jobs = -1
        pathprefix = "HRI-Error-Detection-STAI/"
    print("n_jobs:", n_jobs)

    config = TS_Model_Trainer.read_config(pathprefix+"code/config.json")

    trainer = TS_Model_Trainer(pathprefix+"data/", task=2, n_jobs=n_jobs)

    trainer.data.limit_to_sessions(
        sessions_train=[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], sessions_val=[0, 3, 4, 7, 8, 11, 12, 13])

    study = trainer.optuna_study(
        n_trials=config["n_trials"], model_type=config["model_type"], study_name=config["model_type"], verbose=True)

    if False:
        for col in trainer.data.train_X.columns:
            print(col)

        print(len(trainer.data.train_X), len(trainer.data.val_X))

        # print labels of val where "session" is 0
        print(trainer.data.val_Y[trainer.data.val_Y['session'] == 0])
        # print counts for "UserAwkwardness" in val where "session" is 0, from the first 100 rows
        print(trainer.data.val_Y[trainer.data.val_Y['session'] == 0]
              ['UserAwkwardness'].value_counts()[:1000])
        print(trainer.data.val_Y[trainer.data.val_Y['session'] == 0]
              ['RobotMistake'].value_counts()[:1000])

        val_X_TS, val_Y_TS, train_X_TS, train_Y_TS = trainer.data.get_timeseries_format(
            intervallength=1000, stride=1000, verbose=True)  # 1000 corresponds to 10 seconds(because 100fps)

        print("Val shape", val_X_TS.shape, val_Y_TS.shape,
              "Train shape", train_X_TS.shape, train_Y_TS.shape)
        print(val_Y_TS)

        # get first column of val_Y_TS
        task = 2
        train_Y_TS_task = train_Y_TS[:, task]
        val_Y_TS_task = val_Y_TS[:, task]

        # Train the model
        print("Training model...")
        model = MINIROCKET.MiniRocketVotingClassifier(n_estimators=4, n_jobs=4)
        model.fit(train_X_TS, train_Y_TS_task)
        test_acc = model.score(val_X_TS, val_Y_TS_task)
        print("Test accuracy:", test_acc)
        test_preds = model.predict(val_X_TS)
        # print counts
        print(pd.Series(test_preds).value_counts())
        # confusion matrix from sklearn
        print(confusion_matrix(val_Y_TS_task, test_preds))
