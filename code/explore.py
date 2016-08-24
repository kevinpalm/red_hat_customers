from utilities import simple_load
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def labelplot(train, test):

    """

    :param train: pandas.DataFrame
        The kaggle training dataset joined with the kaggle people dataset
    :param test:  pandas.DataFrame
        The kaggle testing dataset joined with the kaggle people dataset
    :return:
        None

    """

    # Make a frequency histogram for the output labels
    train["outcome"].plot.hist(bins=2, figsize=(3, 3))
    plt.tight_layout()
    plt.xticks([0.25, 0.75], [0, 1])
    plt.savefig("../images/output_label_hist.png")
    plt.clf()


def typeplot(train, test):

    """

    :param train: pandas.DataFrame
        The kaggle training dataset joined with the kaggle people dataset
    :param test:  pandas.DataFrame
        The kaggle testing dataset joined with the kaggle people dataset
    :return:
        None

    """

    # Create a groupby object for the chart
    df = train.groupby(["activity_category", "outcome"])["activity_id"].count()
    df = df.reset_index()
    df["Outcome Label 0 Frequency"] = (df["outcome"]*-1+1)*df["activity_id"]
    df["Outcome Label 1 Frequency"] = df["outcome"]*df["activity_id"]

    # Plot a bar chart
    df[["Outcome Label 0 Frequency", "Outcome Label 1 Frequency"]].plot.bar(figsize=(9, 3))
    plt.xticks(range(len(df["activity_category"].tolist())), df["activity_category"].tolist())
    plt.tight_layout()
    plt.savefig("../images/output_type_bar.png")
    plt.clf()

def groupplot(train, test):

    """

    :param train: pandas.DataFrame
        The kaggle training dataset joined with the kaggle people dataset
    :param test:  pandas.DataFrame
        The kaggle testing dataset joined with the kaggle people dataset
    :return:
        None

    """

    # Drop unneeded features and add the source
    df = train[["date_act", "group_1", "outcome"]]

    # Read in the benchmark predictions, add them to the testing data and format to match the above df
    predicts = pd.read_csv("../output/benchmark_submission.csv")
    formatpredicts = test[["date_act", "group_1"]].reset_index()
    formatpredicts["prediction"] = predicts["outcome"]

    # Append together
    df = df.append(formatpredicts)

    # Pick out ten random groups who have at least 10 predictions and examples of both types in their training data
    grps = pd.DataFrame()
    grps["Min Label"] = df.groupby("group_1")["prediction"].min()
    grps["Max Label"] = df.groupby("group_1")["prediction"].max()
    grps["Count Label"] = df.groupby("group_1")["prediction"].count()
    grps = grps[grps["Min Label"] < 0.25][grps["Max Label"] > 0.75][grps["Count Label"] >= 10]
    grps = grps.sort_index().sample(5, random_state=42)
    grps = list(grps.index)

    # Prepare graphic objects
    axdict = {}
    f, (axdict["ax1"], axdict["ax2"], axdict["ax3"], axdict["ax4"], axdict["ax5"]) = plt.subplots(5, sharex=True,
                                                                                                  sharey=True)
    # Set up the data for each object
    for ax in axdict.keys():
        subdf = df[df["group_1"] == grps[axdict.keys().index(ax)]]
        subdf["date_act"] = (subdf["date_act"]-subdf["date_act"].min())/np.timedelta64(1, 'D')
        axdict[ax].scatter(subdf["date_act"], subdf["prediction"], color='r', alpha=0.4)
        axdict[ax].scatter(subdf["date_act"], subdf["outcome"], color='b', alpha=0.4)

    # Add a title
    axdict["ax1"].set_title(
        'Days Ongoing vs. Output Labels for Five Randomly Selected Groups', y=1.20)

    # Add the legend to the bottom
    plt.legend(bbox_to_anchor=(0.5, -0.675), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    # make subplots close to each other
    f.subplots_adjust(hspace=0.15)

    # hide x ticks for all but bottom plot
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    # size down y ticks
    plt.setp([a.get_yticklabels() for a in f.axes], fontsize=8)

    # Write to file
    plt.savefig("../images/output_group_scatters.png")
    plt.clf()

def main():
    train, test = simple_load()
    groupplot(train, test)

if __name__ == "__main__":
    main()

