import pandas as pd

def peopleprep():

    # Read in the people dataset
    people = pd.read_csv("../data/act_train.csv")
    print people.head()

def summarize(feature):
    pass

peopleprep()