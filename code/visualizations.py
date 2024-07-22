import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import json
import seaborn as sns
plt.rcParams.update({'font.size': 20})

REMAPPING = {
    'REMOVE_NOTHING': 'All features',
    'openface': 'No OpenFace',
    'openpose': 'No OpenPose',
    'opensmile': 'No OpenSmile',
    'speaker': 'No Speaker Diarization',
    'frame': 'No Frame',
    'only_speaker': 'Speaker Diarization only',
    'only_opensmile': 'OpenSmile only',
    'only_openface': 'OpenFace only',
    'only_openpose': 'OpenPose only',
    'only_frame': 'Frame only'
}

NAIVE_ACC = 0.7596
NAIVE_F1 = 0.43


def plot_feature_importance(runs: list = None, offline=True):
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

    scores = scores_file["scores"]
    scores_mean = scores_file["mean_scores"]
    max_sessions = 55  # number of training files
    stepsize = 3  # stepsize of training files

    scores = np.array(scores)
    # revert order of scores
    scores = scores[::-1]
    scores_mean = np.mean(scores, axis=1)
    print("\n\nMean Scores:", scores_mean)

    # plot learning curve with standard deviation
    plt.figure(figsize=(8, 4))
    start_step = max_sessions % stepsize
    plt.plot(range(start_step, max_sessions+1, stepsize), scores_mean)
    plt.fill_between(range(start_step, max_sessions+1, stepsize), scores_mean -
                     np.std(scores, axis=1), scores_mean + np.std(scores, axis=1), alpha=0.2)
    plt.xlabel("Number of sessions in training data")
    plt.xlim([0, max_sessions+1])
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    # save as pdf in plots folder
    plt.savefig("plots/learning_curve.pdf")
    plt.show()


if __name__ == "__main__":
    plot_feature_importance(offline=True)
    plot_learning_curve()
    violin_plots()
