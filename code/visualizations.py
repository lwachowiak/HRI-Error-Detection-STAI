import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(run):
    feature_importance_hist = run_feature_importance.history()
    feature_importance_hist.head()
    # remove rows where accuracy is NaN
    feature_importance_hist = feature_importance_hist.dropna(subset=[
                                                             'accuracy'])
    # sort by accuracy
    feature_importance_hist = feature_importance_hist.sort_values(
        'accuracy', ascending=True)
    plt.figure(figsize=(10, 6))
    standard_deviation = feature_importance_hist.groupby('columns_to_remove')[
        'accuracy'].std()
    standard_deviation = standard_deviation[feature_importance_hist['columns_to_remove']]
    # sorted by accuracy
    plt.barh(feature_importance_hist['columns_to_remove'],
             feature_importance_hist['accuracy'], xerr=standard_deviation)
    # add restric x-axis to 0.6 to 1
    plt.xlim(0.6, 0.85)
    # add grid with alpha
    plt.grid(alpha=0.25)
    plt.xlabel('Accuracy')
    plt.ylabel('Columns to Remove')
    plt.show()
    # save plot
    plt.savefig('plots/feature_importance.pdf')


def violin_plots(histories):

    def individual_violin_plot(histories, key, x_label):
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
        plt.show()
        # save plot
        plt.savefig(f'plots/{key}_violin_plot.pdf')

    # make one joint df and add model name as column
    for i, hist in enumerate(histories):
        hist['model_name'] = ['Random Forest', 'MiniRocket', 'ConvTran'][i]
    df = pd.concat(histories)
    print(df.columns)
    # drop rows were interval_length is nan
    df = df.dropna(subset=['interval_length'])

    # Creating the violin plot
    individual_violin_plot(df, 'interval_length', 'Interval Length in Seconds')
    individual_violin_plot(df, 'stride_eval', 'Evaluation Stride in Seconds')
    individual_violin_plot(df, 'stride_train', 'Training Stride in Seconds')
    individual_violin_plot(df, 'rescaling', 'With/Without Normalization')


if __name__ == "__main__":
    # feature importance
    api = wandb.Api()
    run_feature_importance = api.run("lennartw/HRI-Errors/9gq8vvou")
    plot_feature_importance(run_feature_importance)

    # violin plots
    run = api.run("lennartw/HRI-Errors/pe0z1db0")  # rf
    run2 = api.run("lennartw/HRI-Errors/nk6xvdk2")  # minirocket
    run3 = api.run("lennartw/HRI-Errors/no3vd1bk")  # convtran
    histories = [run.history(), run2.history(), run3.history()]
    violin_plots(histories)
