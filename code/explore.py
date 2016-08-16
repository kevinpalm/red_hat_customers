from utilities import simpleload
import matplotlib.pyplot as plt


def labelplot():

    # Make a frequency histogram for the output labels
    train["outcome"].plot.hist(bins=2, figsize=(3, 3))
    plt.tight_layout()
    plt.xticks([0.25, 0.75], [0, 1])
    plt.savefig("../images/output_label_hist.png")
    plt.clf()

def activitydateplot(datecol):

    # Make a line graph pertaining to the frequency and the label mean
    group = train.group

train, test = simpleload()

print train.columns.values

print train["date_act"].min()
print train["date_act"].max()

print train["date"].min()
print train["date"].max()

