from time_series_classifiers import TS_Model_Trainer
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", type=str, default="MiniRocket_2024-07-09-12.json")
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

for i in range(4):
    trainer.train_and_save_best_model(
        args.file,
        fold=i)
    print("NEXT FOLD")
