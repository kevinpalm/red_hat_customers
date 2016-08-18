import pandas as pd
import numpy as np


def simple_load():

    # Read in the data
    people = pd.read_csv("../data/people.csv")
    train = pd.read_csv("../data/act_train.csv")
    test = pd.read_csv("../data/act_test.csv")

    # Merge people to the other data sets
    train = train.merge(people, on="people_id", suffixes=("_act", ""))
    test = test.merge(people, on="people_id", suffixes=("_act", ""))

    # Correct some data types
    for field in ["date_act", "date"]:
        train[field] = pd.to_datetime(train[field])
        test[field] = pd.to_datetime(test[field])

    return train, test

def group_split(group, train, test):

    # Get the subsets
    subtrain = train[train["group_1"] == group]
    subtest = test[test["group_1"] == group]

    # Prep the features
    subtrain_x = pd.DataFrame((subtrain["date_act"]-subtrain["date_act"].min())/np.timedelta64(1, 'D'))
    subtrain_y = subtrain["outcome"]
    subtest_x = pd.DataFrame((subtest["date_act"]-subtest["date_act"].min())/np.timedelta64(1, 'D'))



    return subtrain_x, subtrain_y, subtest_x