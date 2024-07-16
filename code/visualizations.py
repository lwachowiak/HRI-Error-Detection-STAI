import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

try:
    import wandb
    import seaborn as sns
    offline = False
except ImportError:
    offline = True
    print("Seaborn and wandb not installed, running offline")

REMAPPING = {
    'REMOVE_NOTHING': 'All Features',
    'openface': 'No OpenFace',
    'openpose': 'No OpenPose',
    'opensmile': 'No OpenSmile',
    'speaker': 'No Speaker Diarization',
    'frame': 'No Frame',
    'only_openface': 'OpenFace Only',
    'only_openpose': 'OpenPose Only',
    'only_opensmile': 'OpenSmile Only',
    'only_speaker': 'Speaker Diarization Only',
    'only_frame': 'Frame Only',
}

NAIVE_ACC = 0.7596
NAIVE_F1 = 0.43

def plot_feature_importance(runs: list=None, offline=True):
    if offline:
        #find and load all files in data folder that have feature_search in name
        # The files stored in the data folder are already the history dataframes
        files = [f for f in os.listdir('data') if 'features_search' in f]
        histories = [pd.read_pickle(f'data/{f}') for f in files]
    else:
        histories = [r.history() for r in runs]

    # remove rows where accuracy is NaN
    histories = [h.dropna(subset=['accuracy']) for h in histories]
    histories = [h[['columns_to_remove', 'accuracy', 'macro f1']] for h in histories]

    grouped_hists = [h.groupby('columns_to_remove').agg({
        'accuracy': ['mean', 'std'],
        'macro f1': ['mean', 'std']}).reset_index() for h in histories]
    
    # which index is every dict key? for each key, get the index
    keys = {key: i for i, key in enumerate(grouped_hists[0]['columns_to_remove'])}
    
    idx = []
    for key in REMAPPING.keys():
        idx.append(keys[key])

    idx.reverse()
    
    # sort by the remapping dictionary, 1st key is 1st row, 2nd key is 2nd row etc.
    grouped_hists = [h.reindex(idx) for h in grouped_hists]
    
    # replace columns_to_remove with human readable names and subtract naive baseline
    for h in grouped_hists:
        h['columns_to_remove'] = h['columns_to_remove'].apply(lambda x: REMAPPING[x])
        # subtract naive baseline
        h[('accuracy', 'mean')] = h[('accuracy', 'mean')].apply(lambda x: x - NAIVE_ACC)
        h[('macro f1', 'mean')] = h[('macro f1', 'mean')].apply(lambda x: x - NAIVE_F1)

    print(grouped_hists)
    # plot accuracy
    y = np.arange(len(h['columns_to_remove']))
    plt.figure(figsize=(12, 6))

    for i, h in enumerate(grouped_hists):
        # TODO: shift the points so different model types are on their own y-axis but under the same categorical variable
        plt.errorbar(h['accuracy']['mean'],
                     y - 0.4 + 0.267 * i, 
                     xerr=h['accuracy']['std'], 
                     fmt='o',
                     markersize=14,
                     elinewidth=4, 
                     label=['Random Forest', 'MiniRocket', 'ConvTranPlus', 'TST'][i],
                     color=['#4a7fa4', '#e1812b', '#3a923a', '#c03d3e'][i]
                     )
    #add shaded areas for every category across the y-axis
    for i in range(1, len(REMAPPING), 2):
        plt.axhspan(i - 0.5, i + 0.5, alpha=0.2, color='grey')
    plt.axvline(x=0., color='black', linestyle='dotted', label='Naive Baseline')
    #rotate labels
    plt.yticks(y, grouped_hists[0]['columns_to_remove'],rotation=45)
    # add restric x-axis to 0.6 to 1
    plt.xlim(-0.13, 0.1)
    plt.legend(title='Model', loc='upper left')
    # add grid with alpha
    plt.grid(alpha=0.25)
    plt.xlabel("Accuracy difference")
    plt.ylabel("Feature groups")
    plt.tight_layout()
    # save plot
    plt.savefig('plots/feature_importance_accuracy.pdf')
    plt.show()

    # plot f1
    plt.figure(figsize=(12, 6))

    for i, h in enumerate(grouped_hists):
        # TODO: shift the points so they are not plotted on the same x-axis
        plt.errorbar(h['macro f1']['mean'], 
                     y - 0.4 + 0.267 * i, 
                     xerr=h['macro f1']['std'], 
                     fmt='o',
                     markersize=14,
                     elinewidth=4, 
                     label=['Random Forest', 'MiniRocket', 'ConvTranPlus', 'TST'][i], 
                     color=['#4a7fa4', '#e1812b', '#3a923a', '#c03d3e'][i]
                     )
    #add shaded areas for every category across the y-axis
    for i in range(1, len(REMAPPING), 2):
        plt.axhspan(i - 0.5, i + 0.5, alpha=0.2, color='grey')
    plt.axvline(x=0., color='black', linestyle='dotted', label='Naive Baseline')
    #rotate labels
    plt.yticks(y,grouped_hists[0]['columns_to_remove'], rotation=45)


    plt.xlim(-0.01, 0.35)
    plt.legend(title='Model', loc='upper left')
    # add grid with alpha
    plt.grid(alpha=0.25)
    plt.xlabel("Macro F1")
    plt.ylabel("Feature Groups")
    plt.tight_layout()
    # save plot
    plt.savefig('plots/feature_importance_f1.pdf')
    plt.show()


def violin_plots(histories: list):

    def individual_violin_plot(histories, key, x_label, xticks):
        sns.violinplot(x=key, y='accuracy', data=histories,
                       hue="model_name", cut=0)
        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        # add legend
        plt.legend(title='Model', loc='lower right')
        # y axis with 2 decimal places
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        # add transparent grid
        plt.grid(alpha=0.25)
        # add xticks list of strings
        plt.xticks(ticks=range(len(xticks)), labels=xticks)
        # save plot
        plt.savefig(f'plots/{key}_violin_plot.pdf')
        plt.show()

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
                           'Interval Length in Seconds', ["5s", "15s", "25s"])
    individual_violin_plot(
        df, 'stride_eval', 'Evaluation Stride in Seconds', ["3s", "6s", "9s"])
    individual_violin_plot(
        df, 'stride_train', 'Training Stride in Seconds', ["3s", "6s", "9s"])
    individual_violin_plot(
        df, 'rescaling', 'With/Without Normalization', ["With", "Without"])


if __name__ == "__main__":
    # feature importance

    if offline:
        plot_feature_importance(offline=True)

    else:
        api = wandb.Api()
        run_feature_importance = api.run("lennartw/HRI-Errors/9gq8vvou")
        plot_feature_importance(run_feature_importance)

        # violin plots
        run = api.run("lennartw/HRI-Errors/pe0z1db0")  # rf
        run2 = api.run("lennartw/HRI-Errors/nk6xvdk2")  # minirocket
        run3 = api.run("lennartw/HRI-Errors/no3vd1bk")  # convtran
        run4 = api.run("lennartw/HRI-Errors/d8r87xzk")  # tst
        histories = [run.history(), run2.history(), run3.history(), run4.history()]
        violin_plots(histories)
