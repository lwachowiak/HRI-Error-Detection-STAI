import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import json
import seaborn as sns
import pickle
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 20})

ALL_COMBINATIONS_REMAPPING = {
    "REMOVE_NOTHING": "All features",
        "openpose": "No OpenPose", 
        "r_openface": "No OF continuous AUs", 
        "c_openface": "No OF binary AUs", 
        "opensmile": "No openSMILE", 
        "speaker": "No Speaker Diarization", 
        "frame": "No Frame",
        "r_openface,c_openface": "No OpenFace", 
        "openpose,r_openface": "No OpenPose and OF continuous AUs", 
        "openpose,c_openface": "No OpenPose and OF binary AUs", 
        "openpose,opensmile": "No OpenPose and openSMILE", 
        "openpose,speaker": "No OpenPose and Speaker Diarization", 
        "openpose,frame": "No OpenPose and Frame", 
        "r_openface,opensmile": "No OF continuous AUs and openSMILE", 
        "r_openface,speaker": "No OF continuous AUs and Speaker Diarization", 
        "r_openface,frame": "No OF continuous AUs and Frame", 
        "c_openface,opensmile": "No OF binary AUs and openSMILE", 
        "c_openface,speaker": "No OF binary AUs and Speaker Diarization", 
        "c_openface,frame": "No OF binary AUs and Frame", 
        "opensmile,speaker": "No openSMILE and Speaker Diarization", 
        "opensmile,frame": "No openSMILE and Frame", 
        "speaker,frame": "No Speaker Diarization and Frame", 
        "openpose,r_openface,c_openface": "No OpenPose and OpenFace", 
        "openpose,r_openface,opensmile": "No OpenPose, OF continuous AUs and openSMILE",
        "openpose,r_openface,speaker": "No OpenPose, OF continuous AUs and Speaker Diarization",
        "openpose,r_openface,frame": "No OpenPose, OF continuous AUs and Frame",
        "openpose,c_openface,opensmile": "No OpenPose, OF binary AUs and openSMILE",
        "openpose,c_openface,speaker": "No OpenPose, OF binary AUs and Speaker Diarization",
        "openpose,c_openface,frame": "No OpenPose, OF binary AUs and Frame",
        "openpose,opensmile,speaker": "No OpenPose, openSMILE and Speaker Diarization",
        "openpose,opensmile,frame": "No OpenPose, openSMILE and Frame",
        "openpose,speaker,frame": "No OpenPose, Speaker Diarization and Frame",
        "r_openface,c_openface,opensmile": "No OF continuous AUs, OF binary AUs and openSMILE",
        "r_openface,c_openface,speaker": "No OF continuous AUs, OF binary AUs and Speaker Diarization",
        "r_openface,c_openface,frame": "No OF continuous AUs, OF binary AUs and Frame",
        "r_openface,opensmile,speaker": "No OF continuous AUs, openSMILE and Speaker Diarization",
        "r_openface,opensmile,frame": "No OF continuous AUs, openSMILE and Frame",
        "r_openface,speaker,frame": "No OF continuous AUs, Speaker Diarization and Frame",
        "c_openface,opensmile,speaker": "No OF binary AUs, openSMILE and Speaker Diarization",
        "c_openface,opensmile,frame": "No OF binary AUs, openSMILE and Frame",
        "c_openface,speaker,frame": "No OF binary AUs, Speaker Diarization and Frame",
        "opensmile,speaker,frame": "OpenPose and OpenFace",
        "openpose,r_openface,c_openface,opensmile": "Speaker Diarization and Frame",
        "openpose,r_openface,c_openface,speaker": "openSMILE and Frame",
        "openpose,r_openface,c_openface,frame": "openSMILE and Speaker Diarization",
        "openpose,r_openface,opensmile,speaker": "OF binary AUs and Frame",
        "openpose,r_openface,opensmile,frame": "OF binary AUs and Speaker Diarization",
        "openpose,r_openface,speaker,frame": "OF binary AUs and openSMILE",
        "openpose,c_openface,opensmile,speaker": "OF continuous AUs and Frame",
        "openpose,c_openface,opensmile,frame": "OF continuous AUs and Speaker Diarization",
        "openpose,c_openface,speaker,frame": "OF continuous AUs and openSMILE",
        "openpose,opensmile,speaker,frame": "OpenFace only",
        "r_openface,c_openface,opensmile,speaker": "OpenPose and Frame",
        "r_openface,c_openface,opensmile,frame": "OpenPose and Speaker Diarization",
        "r_openface,c_openface,speaker,frame": "OpenPose and openSMILE",
        "r_openface,opensmile,speaker,frame": "OF binary AUs and OpenPose",
        "c_openface,opensmile,speaker,frame": "OF continuous AUs and OpenPose",
        "openpose,r_openface,c_openface,opensmile,speaker": "Frame only",
        "openpose,r_openface,c_openface,opensmile,frame": "Speaker Diarization only",
        "openpose,r_openface,c_openface,speaker,frame": "openSMILE only",
        "openpose,r_openface,opensmile,speaker,frame": "OF binary AUs only",
        "openpose,c_openface,opensmile,speaker,frame": "OF continuous AUs only",
        "r_openface,c_openface,opensmile,speaker,frame": "OpenPose only"
}

REMAPPING = {
    'REMOVE_NOTHING': 'All features',
    'openface': 'No OpenFace',
    'openpose': 'No OpenPose',
    'opensmile': 'No openSMILE',
    'speaker': 'No Speaker Diarization',
    'frame': 'No Frame',
    'openpose, c_openface': 'No pose and binary AUs',
    'only_speaker': 'Speaker Diarization only',
    'only_opensmile': 'openSMILE only',
    'only_openface': 'OpenFace only',
    'only_openpose': 'OpenPose only',
    'only_frame': 'Frame only'
}

NAME_REMAPPING = {
    "minirocket": "MiniRocket",
    "rf": "Random Forest",
    "convtran": "ConvTran",
    "tst": "TST"
}

NAIVE_ACC = 0.7596
NAIVE_F1 = 0.43

def plot_feature_importance(plot: bool = True, save: bool = True, horizontal: bool = False) -> None:
    """
    method for interpretability of RFs
    """
    if os.getcwd().endswith("HRI-Error-Detection-STAI"):
        pathprefix = ""
    else:
        pathprefix = "HRI-Error-Detection-STAI/"

    model_to_load = "RandomForest"

    with open(pathprefix + "code/trained_models/" + model_to_load + ".pkl", "rb") as f:
        model = pickle.load(f)

    with open(pathprefix + "code/trained_models/" + model_to_load + "_columns.pkl", "rb") as f:
        feature_names = pickle.load(f)

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

        blue_patch = mpatches.Patch(color='blue', label='OpenPose')
        orange_patch = mpatches.Patch(color='orange', label='OpenFace')
        green_patch = mpatches.Patch(color='green', label='OpenSMILE')
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

def plot_feature_groups_performance(runs: list = None, offline=True):
    if offline:
        # find and load all files in data folder that have feature_search in name
        # The files stored in the data folder are already the history dataframes
        files = [f for f in os.listdir(
            'plots/run_histories') if 'features_search' in f]
        histories = []
        for f in files:
            p = pd.read_pickle(f'plots/run_histories/{f}')
            if 'tst' in f:
                pass
            elif 'convtran' in f:
                pass
            else:
                histories.append(p)
        # join convtran and tst histories
        convtran = [pd.read_pickle(f'plots/run_histories/{f}')
                    for f in files if 'convtran' in f]
        tst = [pd.read_pickle(
            f'plots/run_histories/{f}') for f in files if 'tst' in f]
        print("ConvTran runs:", len(convtran))
        print("TST runs:", len(tst))
        convtran = [pd.concat(convtran)]
        tst = [pd.concat(tst)]

        histories = histories + convtran + tst

    else:
        histories = [r.history() for r in runs]

    # remove rows where accuracy is NaN
    histories = [h.dropna(subset=['accuracy']) for h in histories]
    histories = [h[['columns_to_remove', 'accuracy', 'macro f1']]
                 for h in histories]

    grouped_hists = [h.groupby('columns_to_remove').agg({
        'accuracy': ['mean', 'std'],
        'macro f1': ['mean', 'std']}).reset_index() for h in histories]

    # which index is every dict key? for each key, get the index
    keys = {key: i for i, key in enumerate(
        grouped_hists[0]['columns_to_remove'])}

    idx = []
    for key in REMAPPING.keys():
        idx.append(keys[key])

    idx.reverse()

    # sort by the remapping dictionary, 1st key is 1st row, 2nd key is 2nd row etc.
    grouped_hists = [h.reindex(idx) for h in grouped_hists]

    # replace columns_to_remove with human readable names and subtract naive baseline
    for h in grouped_hists:
        h['columns_to_remove'] = h['columns_to_remove'].apply(
            lambda x: REMAPPING[x])
        # subtract naive baseline
        h[('accuracy', 'mean')] = h[('accuracy', 'mean')].apply(
            lambda x: x - NAIVE_ACC)
        h[('macro f1', 'mean')] = h[('macro f1', 'mean')].apply(
            lambda x: x - NAIVE_F1)

    print(grouped_hists)
    # plot accuracy
    y = np.arange(len(h['columns_to_remove']))
    plt.figure(figsize=(10, 8))

    for i, h in enumerate(grouped_hists):
        # TODO: shift the points so different model types are on their own y-axis but under the same categorical variable
        plt.errorbar(h['accuracy']['mean'],
                     y - 0.35 + 0.233 * i,
                     xerr=h['accuracy']['std'],
                     fmt='o',
                     markersize=14,
                     elinewidth=4,
                     label=['Random Forest', 'MiniRocket',
                            'ConvTran', 'TST'][i],
                     color=['#4a7fa4', '#e1812b', '#3a923a', '#c03d3e'][i]
                     )
    # add shaded areas for every category across the y-axis
    for i in range(1, len(REMAPPING), 2):
        plt.axhspan(i - 0.5, i + 0.5, alpha=0.2, color='grey')
    plt.axvline(x=0., color='black', linestyle='dotted',
                label='Naive Baseline')
    # rotate labels
    plt.yticks(y, grouped_hists[0]['columns_to_remove'], rotation=45)
    # add restric x-axis to 0.6 to 1
    plt.xlim(-0.13, 0.1)
    plt.legend(title='Model', loc='upper left', prop={'size': 14})
    # add grid with alpha
    plt.grid(alpha=0.25)
    plt.xlabel("Accuracy difference")
    plt.ylabel("Feature groups")
    plt.tight_layout()
    # save plot
    plt.savefig('plots/feature_importance_accuracy.pdf')
    plt.show()

    # plot f1
    plt.figure(figsize=(10, 8))

    for i, h in enumerate(grouped_hists):
        # TODO: shift the points so they are not plotted on the same x-axis
        plt.errorbar(h['macro f1']['mean'],
                     y - 0.35 + 0.233 * i,
                     xerr=h['macro f1']['std'],
                     fmt='o',
                     markersize=14,
                     elinewidth=4,
                     label=['Random Forest', 'MiniRocket',
                            'ConvTran', 'TST'][i],
                     color=['#4a7fa4', '#e1812b', '#3a923a', '#c03d3e'][i]
                     )
    # add shaded areas for every category across the y-axis
    for i in range(1, len(REMAPPING), 2):
        plt.axhspan(i - 0.5, i + 0.5, alpha=0.2, color='grey')
    plt.axvline(x=0., color='black', linestyle='dotted',
                label='Naive Baseline')
    # rotate labels
    plt.yticks(y, grouped_hists[0]['columns_to_remove'], rotation=45)

    plt.xlim(-0.01, 0.35)
    plt.legend(title='Model', loc='upper left', prop={'size': 14})
    # add grid with alpha
    plt.grid(alpha=0.25)
    plt.xlabel("Macro F1 difference")
    plt.ylabel("Feature Groups")
    plt.tight_layout()
    # save plot
    plt.savefig('plots/feature_importance_f1.pdf')
    plt.show()


def violin_plots():

    def individual_violin_plot(histories, key, x_label, xticks, legend_loc):
        plt.figure(figsize=(10, 5))
        sns.violinplot(x=key, y='accuracy', data=histories,
                       hue="model_name", cut=0, legend=False if legend_loc == None else True)
        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        # add legend
        if legend_loc != None:
            plt.legend(title='Model', loc=legend_loc, prop={'size': 16})

        # y axis with 2 decimal places
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        # add transparent grid
        plt.grid(alpha=0.25)
        # add xticks list of strings
        plt.xticks(ticks=range(len(xticks)), labels=xticks)
        plt.tight_layout()
        # save plot
        plt.savefig(f'plots/{key}_violin_plot.pdf')
        plt.show()

    files = [f for f in os.listdir('plots/run_histories') if 'violin' in f]
    histories = []
    for f in files:
        p = pd.read_pickle(f'plots/run_histories/{f}')
        histories.append(p)
    # make one joint df and add model name as column

    for i, hist in enumerate(histories):
        hist['model_name'] = ['Random Forest',
                              'MiniRocket', 'ConvTran', 'TST'][i]
    df = pd.concat(histories)
    print(df.columns)
    # drop rows were interval_length is nan
    df = df.dropna(subset=['interval_length'])

    # Creating the violin plot
    individual_violin_plot(df, 'interval_length',
                           'Interval Length [s]', ["5", "15", "25"], None)
    individual_violin_plot(
        df, 'stride_eval', 'Evaluation Stride [s]', ["3", "6", "9"], None)
    individual_violin_plot(
        df, 'stride_train', 'Training Stride [s]', ["3", "6", "9"], None)
    individual_violin_plot(
        df, 'rescaling', 'Normalization', ["With", "Without"], 'lower left')


def plot_learning_curve(scores_file: str = "plots/run_histories/learning_curve_study.json"):
    # read the scores file json and load it
    with open(scores_file, 'r') as f:
        scores_file = json.load(f)

    scores = [scores_file[key] for key in scores_file.keys()]
    names = [NAME_REMAPPING[key] for key in scores_file.keys()]
    max_sessions = 55  # number of training files
    stepsize = 3  # stepsize of training files

    scores = [np.array(s) for s in scores]
    scores_mean = [np.mean(s, axis=1) for s in scores]
    print(scores_mean)

    # plot learning curve with standard deviation
    plt.figure(figsize=(8, 4))
    start_step = max_sessions % stepsize
    for i, sc in enumerate(scores_mean):
        plt.plot(range(start_step, max_sessions+1, stepsize), sc, label=names[i])
        plt.fill_between(range(start_step, max_sessions+1, stepsize), 
                     sc - np.std(scores[i], axis=1), 
                     sc + np.std(scores[i], axis=1), 
                     alpha=0.2
                     )
    plt.hlines(NAIVE_ACC, 0, max_sessions, linestyles='dotted', colors='black', label='Naive Baseline', linewidth=3)
    plt.xlabel("Number of sessions in training data")
    plt.xlim([0, max_sessions+1])
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.2)
    plt.legend(title='Model', loc='lower right', prop={'size': 12})
    plt.tight_layout()
    # save as pdf in plots folder
    plt.savefig("plots/learning_curve.pdf")
    plt.show()

def plot_all_features():
    #find file that has all in its name, should only be one
    for f in os.listdir('plots/run_histories'):
        if 'all' in f:
            all_features = pd.read_pickle(f'plots/run_histories/{f}')
            break

    all_features = all_features[['accuracy', 'macro f1','columns_to_remove']]

    # sort by accuracy
    all_features = all_features.sort_values(by='accuracy', ascending=True)

    # plot with the correct labels
    all_features['columns_to_remove'] = all_features['columns_to_remove'].apply(lambda x: ALL_COMBINATIONS_REMAPPING[x])

    # do a big horizontal plot and use the whole A4 page
    plt.figure(figsize=(16, 22))
    y = np.arange(len(all_features))
    plt.errorbar(
        all_features['accuracy'], 
        y, 
        fmt='o',  
        markersize=14, 
        elinewidth=4
        )
    plt.axvline(x=NAIVE_ACC, color='black', linestyle='dotted',
                label='Naive Baseline')

    plt.yticks(y, all_features['columns_to_remove'], rotation=0)
    plt.xlabel('Accuracy')
    plt.ylabel('Feature groups')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig('plots/all_feature_importance.pdf')
    plt.show()

if __name__ == "__main__":
    #plot_all_features()
    #plot_feature_importance(offline=True)
    #plot_learning_curve()
    #violin_plots()
    
    plot_feature_importance()