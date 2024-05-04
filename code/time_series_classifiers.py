from tsai.models import MINIROCKET
from data_loader import DataLoader_HRI
from tsai.data.all import *
from tsai.models.utils import *
from tsai.all import my_setup
import pandas as pd
from sklearn.metrics import confusion_matrix
import optuna

# TODO:
# - optuna
# - true eval on full lenght

# Models to try: HydraMultiRocketPlus


# def (data):
#   # Preprocess the data
#   return data

class TS_Model_Trainer:

    def __init__(self, data_folder, task):
        self.data = DataLoader_HRI(data_folder)
        self.task = task

    def evaluate_model(model, data):
        # Evaluate the model
        return None

    def optuna_objective(self, trial: optuna.Trial):
        stride = trial.suggest_int("stride", low=700, high=1000, step=50)
        max_dilations_per_kernel = trial.suggest_int(
            "max_dilations_per_kernel", low=16, high=64, step=8)
        val_X_TS, val_Y_TS, train_X_TS, train_Y_TS = self.data.get_timeseries_format(
            intervallength=1000, stride=stride, verbose=False)

        train_Y_TS_task = train_Y_TS[:, self.task]
        val_Y_TS_task = val_Y_TS[:, self.task]

        model = MINIROCKET.MiniRocketVotingClassifier(
            n_estimators=2, n_jobs=4, max_dilations_per_kernel=max_dilations_per_kernel)
        model.fit(train_X_TS, train_Y_TS_task)

        test_acc = model.score(val_X_TS, val_Y_TS_task)
        test_preds = model.predict(val_X_TS)
        print(confusion_matrix(val_Y_TS_task, test_preds))

        return test_acc


if __name__ == '__main__':
    my_setup(optuna)

    trainer = TS_Model_Trainer("data/", task=2)

    trainer.data.limit_to_sessions(
        sessions_train=[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], sessions_val=[0, 3, 4, 7, 8, 11, 12, 13])

    study = optuna.create_study(direction="maximize")
    study.optimize(trainer.optuna_objective, n_trials=6)

    print("Best trial:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_slice(study)
    optuna.visualization.plot_parallel_coordinate(study)

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
