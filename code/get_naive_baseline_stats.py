from time_series_classifiers import TS_Model_Trainer
import optuna
from tsai.all import my_setup, platform
import os

my_setup(optuna)
print(platform.platform())
if os.getcwd().endswith("HRI-Error-Detection-STAI"):
    pathprefix = ""
else:
    pathprefix = "HRI-Error-Detection-STAI/"

trainer = TS_Model_Trainer(folder=pathprefix, n_jobs=8,config_name="best_model_configs/RandomForest_Feature_Importance.json")

trainer.get_naive_baseline_stats()