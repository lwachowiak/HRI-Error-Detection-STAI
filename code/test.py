from time_series_classifiers import TS_Model_Trainer
import os

if os.getcwd().endswith("HRI-Error-Detection-STAI"):
        pathprefix = ""
else:
    pathprefix = "HRI-Error-Detection-STAI/"

trainer = TS_Model_Trainer(
    folder=pathprefix, 
    n_jobs=10, #
    config_name="config_rf.json"
    )

trainer.train_and_save_best_model("RandomForest_2024-06-05-17.json")