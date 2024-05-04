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


def evaluate_model(model, data):
    # Evaluate the model
    return None


def optuna_objective(trial: optuna.Trial):
    stride_space = trial.suggest_int("stride", 100, 1000)

    return None


if __name__ == '__main__':
    my_setup(optuna)

    # Load the data
    data_hri = DataLoader_HRI("data/")

    # Preprocess the data
    # print (column names)
    for col in data_hri.train_X.columns:
        print(col)

    data_hri.limit_to_sessions(
        sessions_train=[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], sessions_val=[0, 3, 4, 7, 8, 11, 12, 13])

    print(len(data_hri.train_X), len(data_hri.val_X))

    # print labels of val where "session" is 0
    print(data_hri.val_Y[data_hri.val_Y['session'] == 0])
    # print counts for "UserAwkwardness" in val where "session" is 0, from the first 100 rows
    print(data_hri.val_Y[data_hri.val_Y['session'] == 0]
          ['UserAwkwardness'].value_counts()[:1000])
    print(data_hri.val_Y[data_hri.val_Y['session'] == 0]
          ['RobotMistake'].value_counts()[:1000])

    val_X_TS, val_Y_TS, train_X_TS, train_Y_TS = data_hri.get_timeseries_format(
        intervallength=1000, stride=100, verbose=True)  # 1000 corresponds to 10 seconds(because 100fps)

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
