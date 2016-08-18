from utilities import simple_load
import matplotlib.pyplot as plt
import pandas as pd


def labelplot():

    # Make a frequency histogram for the output labels
    train["outcome"].plot.hist(bins=2, figsize=(3, 3))
    plt.tight_layout()
    plt.xticks([0.25, 0.75], [0, 1])
    plt.savefig("../images/output_label_hist.png")
    plt.clf()


def typeplot():

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

def groupplot():

    # Drop unneeded features and add the source
    df = train[["date_act", "group_1", "outcome"]]

    # Read in the benchmark predictions, add them to the testing data and format to match the above df
    predicts = pd.read_csv("../output/benchmark_submission.csv")
    formatpredicts = test[["date_act", "group_1"]]
    formatpredicts["prediction"] = predicts["outcome"]

    # Append together
    df = df.append(formatpredicts)

    # Pick out ten random groups who have both types of outcomes in their training data
    grps = pd.DataFrame()
    grps["Min Label"] = df.groupby("group_1")["outcome"].min()
    grps["Max Label"] = df.groupby("group_1")["outcome"].max()
    grps = grps[grps["Min Label"] != grps["Max Label"]]
    grps = grps.sample(10, random_state=42)
    grps = list(grps.index)

    print grps



train, test = simple_load()
groupplot()

