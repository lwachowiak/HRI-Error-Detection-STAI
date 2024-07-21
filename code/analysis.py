import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import pickle
import matplotlib.patches as mpatches


class Analysis:

    def __init__(self):
        pass

    @staticmethod
    def get_impurity_feature_importance(model: object, feature_names: list, plot: bool = True, save: bool = True, horizontal: bool = True) -> None:
        """
        method for interpretability of RFs
        """

        importances = model.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in model.estimators_], axis=0)

        forest_importances = pd.Series(importances, index=feature_names)
        forest_importances = pd.concat(
            [forest_importances, pd.Series(std, index=feature_names)], axis=1)
        forest_importances.columns = ["importance", "std"]
        forest_importances = forest_importances.sort_values(
            by="importance", ascending = False if horizontal else True)

        if plot:
            fig = plt.figure(figsize=(24, 10)) if horizontal else plt.figure(figsize=(10, 24))
            #make the font size bigger for all labels and axes titles
            plt.rcParams.update({'font.size': 17})
            # only plot the feature importances, and use std as error bars
            # change bar color depending on what dataset the feature belongs to: blue = openpose, orange = openface, green = opensmile, yellow = speaker diarization, red = frame
            color = []
            for feature in forest_importances.index:
                if "openpose" in feature:
                    color.append("blue")
                elif "openface" in feature:
                    color.append("orange")
                elif "opensmile" in feature:
                    color.append("green")
                elif "speaker" in feature:
                    color.append("purple")
                elif "frame" in feature:
                    color.append("red")
                else:
                    color.append("black")

            # edit index to drop the _open* suffix
            forest_importances.index = forest_importances.index.str.replace(
                "_openpose", "")
            forest_importances.index = forest_importances.index.str.replace(
                "_openface", "")
            forest_importances.index = forest_importances.index.str.replace(
                "_opensmile", "")

            blue_patch = mpatches.Patch(color='blue', label='Openpose')
            orange_patch = mpatches.Patch(color='orange', label='Openface')
            green_patch = mpatches.Patch(color='green', label='Opensmile')
            purple_patch = mpatches.Patch(color='purple', label='Speaker diarization')
            red_patch = mpatches.Patch(color='red', label='Frame')

            if horizontal:
                plt.grid(axis='both', linestyle='--', alpha=0.5)
                plt.bar(forest_importances.index, forest_importances["importance"], yerr=forest_importances["std"], color=color)
                plt.ylabel("Decrease in impurity")
                plt.xticks(forest_importances.index, rotation=90)
                plt.xlim([-1, len(forest_importances)])
                plt.legend(handles=[blue_patch, orange_patch, green_patch, purple_patch, red_patch], prop={'size': 8}, title="Feature group")

            else:
                plt.grid(axis='both', linestyle='--', alpha=0.5)
                plt.barh(forest_importances.index, forest_importances["importance"], xerr=forest_importances["std"], color=color)
                plt.xlabel("Decrease in impurity")
                plt.yticks(forest_importances.index)
                plt.ylim([-1, len(forest_importances)])
                plt.legend(handles=[blue_patch, orange_patch, green_patch, purple_patch, red_patch], loc='lower right', prop={'size': 20}, title="Feature group")
            
            # ensure the labels are readable
            plt.tight_layout()
            if save:
                fig.savefig("plots/rf_feature_importances_horizonal.pdf" if horizontal else "plots/rf_feature_importances_vertical.pdf")
            else:
                plt.show()


if __name__ == "__main__":

    if os.getcwd().endswith("HRI-Error-Detection-STAI"):
            pathprefix = ""
    else:
        pathprefix = "HRI-Error-Detection-STAI/"

    model_to_load = "RandomForest"

    with open(pathprefix + "code/trained_models/" + model_to_load + ".pkl", "rb") as f:
        model = pickle.load(f)

    with open(pathprefix + "code/trained_models/" + model_to_load + "_columns.pkl", "rb") as f:
        features = pickle.load(f)


    r = Analysis.get_impurity_feature_importance(model, features)
    r2 = Analysis.get_impurity_feature_importance(model, features, horizontal=False)
