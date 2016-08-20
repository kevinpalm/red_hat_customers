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

    # Set index to activity id
    train = train.set_index("activity_id")
    test = test.set_index("activity_id")

    # Correct some data types
    for field in ["date_act", "date"]:
        train[field] = pd.to_datetime(train[field])
        test[field] = pd.to_datetime(test[field])

    return train, test


def extract_leak_features(train, test):

    # Drop the extra columns
    train = train[["group_1", "date_act", "outcome"]]
    test = test[["group_1", "date_act"]]

    # Append the two together for creating features and sort by date and group
    df = train.append(test).sort_values(by=["group_1", "date_act"])

    # Copy train onto the merged index, then shift a copy one step each way for neighbor features, fill accordingly
    indextrain = pd.DataFrame(index=df.index).join(train)
    df = df.join(indextrain.shift(-1), rsuffix="_fbw").join(indextrain.shift(1), rsuffix="_ffw")
    df[["group_1_fbw", "date_act_fbw", "outcome_fbw"]] = df[["group_1_fbw", "date_act_fbw", "outcome_fbw"]].fillna(
        method="bfill")
    df[["group_1_ffw", "date_act_ffw", "outcome_ffw"]] = df[["group_1_ffw", "date_act_ffw", "outcome_ffw"]].fillna(
        method="ffill")

    # Set anything that was filled between groups as nan
    df.loc[(df["group_1"] != df["group_1_ffw"]), ["group_1_ffw", "date_act_ffw", "outcome_ffw"]] = np.nan
    df.loc[(df["group_1"] != df["group_1_fbw"]), ["group_1_fbw", "date_act_fbw", "outcome_fbw"]] = np.nan

    # We want to know the average density for each group
    lookup = pd.DataFrame()
    lookup["min_date"] = df.groupby("group_1")["date_act"].min()
    lookup["max_date"] = df.groupby("group_1")["date_act"].max()
    lookup["range"] = (lookup["max_date"] - lookup["min_date"])/np.timedelta64(1, 'D')
    lookup["data_count"] = df.groupby("group_1")["date_act"].count()
    lookup["density"] = lookup["data_count"]/lookup["range"]
    lookup = lookup[["density"]].reset_index()

    # Merge in density
    df = pd.merge(df.reset_index(), lookup, how="left", on=["group_1"]).set_index("activity_id")

    # Groupby to calculate the mean label for the same dates in the same group
    lookup = train.groupby(["group_1", "date_act"], as_index=False)["outcome"].mean()
    lookup.columns = ["group_1", "date_act", "mean_outcome"]

    # Merge those means into the original sets
    df = pd.merge(df.reset_index(), lookup, how="left", on=["group_1", "date_act"]).set_index("activity_id")

    # Get distance to each side, as a ratio of the density... weird math units but I think it will work okay as a
    # measure of the left and right outcome label dependability
    df["left_distance_ratio"] = (df["date_act"]-df["date_act_ffw"])/np.timedelta64(1, 'D')/df["density"]
    df["right_distance_ratio"] = (df["date_act_fbw"]-df["date_act"])/np.timedelta64(1, 'D')/df["density"]

    # Get values to each side
    df["left_outcome"] = df["outcome_fbw"]
    df["right_outcome"] = df["outcome_ffw"]

    # Replace infinities from the groups with 0 distance with nans
    df = df.replace(np.inf, np.nan)

    # Fill nans in outcome with mean outcome for those testing samples with definite leaks
    df["outcome"] = df["outcome"].fillna(df["mean_outcome"])

    # Resplit the data now that we have more training data
    train = df[pd.notnull(df["outcome"])]
    test = df[pd.isnull(df["outcome"])]

    # Select out features for final output
    # Also fill NaNs with -1... it should be okay because we're going to use a gradient boosting estimator
    train_x = train[["left_distance_ratio", "right_distance_ratio", "left_outcome", "right_outcome"]].fillna(-1.0)
    train_y = train["outcome"]
    test_x = test[["left_distance_ratio", "right_distance_ratio", "left_outcome", "right_outcome"]].fillna(-1.0)


    return train_x, train_y, test_x