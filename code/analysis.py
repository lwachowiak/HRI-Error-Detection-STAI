import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import pickle


class Analysis:

    def __init__(self):
        pass

    @staticmethod
    def get_impurity_feature_importance(model: object, feature_names: list, plot: bool = True, save: bool = False) -> None:
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
            by="importance", ascending=False)
        print(forest_importances)

        if plot:
            fig, ax = plt.subplots()
            # only plot the feature importances, and use std as error bars
            forest_importances["importance"].plot.bar(
                yerr=forest_importances["std"], ax=ax)
            ax.set_title("Feature importances using MDI")
            ax.set_ylabel("Mean decrease in impurity")
            fig.tight_layout()
            plt.show()
            if save:
                fig.savefig("rf_feature_importances.pdf")


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
