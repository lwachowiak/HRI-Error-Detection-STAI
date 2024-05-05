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
import os
import datetime

# TODO:
# - true eval on full lenght
# - just train with some columns / column selection
# - try different models
# - train through CREATE

# Models to try: HydraMultiRocketPlus


# def (data):
#   # Preprocess the data
#   return data

class TS_Model_Trainer:

    def __init__(self, data_folder, task, n_jobs):
        self.data = DataLoader_HRI(data_folder)
        self.task = task
        self.n_jobs = n_jobs

    def evaluate_model(model, data):
        # Evaluate the model
        return None

    def optuna_objective_minirocket(self, trial: optuna.Trial):
        stride = trial.suggest_int("stride", low=500, high=1000, step=50)

        # parameters being optimized
        max_dilations_per_kernel = trial.suggest_int(
            "max_dilations_per_kernel", low=16, high=128, step=8)
        val_X_TS, val_Y_TS, train_X_TS, train_Y_TS = self.data.get_timeseries_format(
            intervallength=1000, stride=stride, verbose=False)
        n_estimators = trial.suggest_int("n_estimators", low=2, high=8, step=2)

        train_Y_TS_task = train_Y_TS[:, self.task]
        val_Y_TS_task = val_Y_TS[:, self.task]

        model = MINIROCKET.MiniRocketVotingClassifier(
            n_estimators=n_estimators, n_jobs=self.n_jobs, max_dilations_per_kernel=max_dilations_per_kernel)
        model.fit(train_X_TS, train_Y_TS_task)

        test_acc = model.score(val_X_TS, val_Y_TS_task)
        test_preds = model.predict(val_X_TS)
        print(confusion_matrix(val_Y_TS_task, test_preds))

        return test_acc

    def optuna_study(self, n_trials, study_name, trainer, verbose=False):

        wandb_kwargs = {"project": "HRI-Errors"}
        wandbc = WeightsAndBiasesCallback(
            metric_name="accuracy", wandb_kwargs=wandb_kwargs)

        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        study = optuna.create_study(
            direction="maximize", study_name=study_name + "_"+date)
        print(f"Sampler is {study.sampler.__class__.__name__}")
        study.optimize(self.optuna_objective_minirocket,
                       n_trials=n_trials, callbacks=[wandbc])

        if verbose:
            print("Best trial:")
            print("Value: ", study.best_trial.value)
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")

        return study


if __name__ == '__main__':
    my_setup(optuna)
    print(os.uname())
    if "Linux" in os.uname():
        n_jobs = -1
    else:
        n_jobs = 2

    trainer = TS_Model_Trainer("data/", task=2, n_jobs=n_jobs)

    trainer.data.limit_to_sessions(
        sessions_train=[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], sessions_val=[0, 3, 4, 7, 8, 11, 12, 13])

    study = trainer.optuna_study(
        n_trials=2, study_name="MiniRocket", trainer=trainer, verbose=True)

    if False:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
        fig = optuna.visualization.plot_param_importances(study)
        fig.show()
        fig = optuna.visualization.plot_slice(study)
        fig.show()
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.show()

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
