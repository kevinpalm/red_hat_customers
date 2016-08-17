from utilities import simpleload
import matplotlib.pyplot as plt


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


train, test = simpleload()
for column in train.columns.values:
    if "char_" in column:
        print column
        print train[column].unique()

