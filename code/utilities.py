import pandas as pd

def simpleload():

    # Read in the data
    people = pd.read_csv("../data/people.csv")
    train = pd.read_csv("../data/act_train.csv")
    test = pd.read_csv("../data/act_test.csv")

    # Join people to the other data sets
    train = train.join(people, on="people_id", lsuffix="_act", rsuffix="_people")
    test = test.join(people, on="people_id", lsuffix="_act", rsuffix="_people")

    return train, test

def peopleprep():

    # Read in the people dataset
    people = pd.read_csv("../data/act_train.csv")
    print people.head()

def summarize(feature):
    pass

peopleprep()