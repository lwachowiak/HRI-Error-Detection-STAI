from time_series_classifiers import TS_Model_Trainer
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", type=str, default="MiniRocket_2024-07-18-06.json")
args = argparser.parse_args()

if os.getcwd().endswith("HRI-Error-Detection-STAI"):
    pathprefix = ""
else:
    pathprefix = "HRI-Error-Detection-STAI/"

print(args.file)

trainer = TS_Model_Trainer(
    folder=pathprefix,
    n_jobs=4,
    config_name="best_model_configs/" + args.file
)

trainer.cross_val_model(config_name=args.file)
