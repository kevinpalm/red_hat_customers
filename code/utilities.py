import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA, RandomizedPCA


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


def group_decision(train, test):
    # Exploit the leak revealed by Loiso and team to try and directly infer any labels that can be inferred
    # https://www.kaggle.com/c/predicting-red-hat-business-value/forums/t/22807/0-987-kernel-now-available-seems-like-leakage

    # Make a lookup dataframe, and copy those in first since we can be sure of them
    lookup = train.groupby(["group_1", "date_act"], as_index=False)["outcome"].mean()
    test = pd.merge(test.reset_index(), lookup, how="left", on=["group_1", "date_act"]).set_index("activity_id")

    # Create some date filling columns that we'll use after we append
    train["date_act_fillfw"] = train["date_act"]
    train["date_act_fillbw"] = train["date_act"]

    # Create some group filling columns for later use
    train["group_fillfw"] = train["group_1"]
    train["group_fillbw"] = train["group_1"]

    # Put the two data sets together and sort
    df = train.append(test)
    df = df.sort_values(by=["group_1", "date_act"])

    # Fill the dates
    df["date_act_fillfw"] = df["date_act_fillfw"].fillna(method="ffill")
    df["date_act_fillbw"] = df["date_act_fillbw"].fillna(method="bfill")

    # Fill labels
    df["outcome_fillfw"] = df["outcome"].fillna(method="ffill")
    df["outcome_fillbw"] = df["outcome"].fillna(method="bfill")

    # Fill the groups
    df["group_fillfw"] = df["group_fillfw"].fillna(method="ffill")
    df["group_fillbw"] = df["group_fillbw"].fillna(method="bfill")

    # Create int booleans for whether the fillers are from the same date
    df["fw_same_date"] = (df["date_act_fillfw"] == df["date_act"]).astype(int)
    df["bw_same_date"] = (df["date_act_fillbw"] == df["date_act"]).astype(int)

    # Create int booleans for whether the fillers are in the same group
    df["fw_same_group"] = (df["group_fillfw"] == df["group_1"]).astype(int)
    df["bw_same_group"] = (df["group_fillbw"] == df["group_1"]).astype(int)

    # Use the filled labels only if the labels were from the same group, unless we're at the end of the group
    df["interfill"] = (df["outcome_fillfw"] *
                       df["fw_same_group"] +
                       df["outcome_fillbw"] *
                       df["bw_same_group"]) / (df["fw_same_group"] +
                                               df["bw_same_group"])

    # If the labels are at the end of the group, cushion by 0.5
    df["needs cushion"] = (df["fw_same_group"] * df["bw_same_group"] - 1).abs()
    df["cushion"] = df["needs cushion"] * df["interfill"] * -0.1 + df["needs cushion"] * 0.05
    df["interfill"] = df["interfill"] + df["cushion"]

    # Fill everything
    df["outcome"] = df["outcome"].fillna(df["interfill"])

    # Return outcomes to the original index
    test["outcome"] = df["outcome"]

    return test["outcome"]


def extract_leak_features(train, test):

    # Drop the extra columns
    train = train[["group_1", "date_act", "outcome"]]
    test = test[["group_1", "date_act"]]

    # Append the two together for creating features and sort by date and group
    df = train.append(test).sort_values(by=["group_1", "date_act"])

    # Copy train onto the merged index, then shift a copy one step each way for neighbor features, fill accordingly
    indextrain = pd.DataFrame(index=df.index).join(train)
    df = df.join(indextrain.shift(-1), rsuffix="_bfill").join(indextrain.shift(1), rsuffix="_ffill")
    df[["group_1_bfill", "date_act_bfill", "outcome_bfill"]] =\
        df[["group_1_bfill", "date_act_bfill", "outcome_bfill"]].fillna(method="bfill")
    df[["group_1_ffill", "date_act_ffill", "outcome_ffill"]] =\
        df[["group_1_ffill", "date_act_ffill", "outcome_ffill"]].fillna(method="ffill")

    # Set anything that was filled between groups as nan
    df.loc[(df["group_1"] != df["group_1_ffill"]), ["group_1_ffill", "date_act_ffill", "outcome_ffill"]] = np.nan
    df.loc[(df["group_1"] != df["group_1_bfill"]), ["group_1_bfill", "date_act_bfill", "outcome_bfill"]] = np.nan

    # We want to know the range and average density for each group
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

    # Get distance to each side as a proportion of the group density... weird math units but seems to work
    df["left_distance"] = (df["date_act"]-df["date_act_ffill"])/np.timedelta64(1, 'D')/df["density"]
    df["right_distance"] = (df["date_act_bfill"]-df["date_act"])/np.timedelta64(1, 'D')/df["density"]

    # Get values to each side
    df["left_outcome"] = df["outcome_bfill"]
    df["right_outcome"] = df["outcome_ffill"]

    # Replace infinities from the groups with 0 distance with nans
    df = df.replace(np.inf, np.nan)

    # Fill nans in outcome with mean outcome for those testing samples with definite leaks
    df["outcome"] = df["outcome"].fillna(df["mean_outcome"])

    # Resplit the data now that we have more training data
    train = df[pd.notnull(df["outcome"])]
    test = df[pd.isnull(df["outcome"])]

    # Select out features for final output
    # Also fill NaNs with -1... it should be okay because we're going to use a gradient boosting estimator anyway
    train_x = train[["left_distance", "right_distance", "left_outcome", "right_outcome"]].fillna(-1.0)
    train_y = train["outcome"]
    test_x = test[["left_distance", "right_distance", "left_outcome", "right_outcome"]].fillna(-1.0)

    return train_x, train_y, test_x

def prep_features(train, test, extra_outcomes=None):

    if isinstance(extra_outcomes, pd.Series):
        # Use the extra known outcomes to resplit the testing data
        test["outcome"] = extra_outcomes
        train = train.append(test[pd.notnull(test["outcome"])])
        test = test[pd.isnull(test["outcome"])].drop("outcome", axis=1)

    # Initialize two empty data frames with matching indexes for storing features
    train_feats = pd.DataFrame(index=train.index)
    test_feats = pd.DataFrame(index=test.index)

    # Seasonal features for activity date, we only need <1 year types because that's all the training data includes
    train_feats = train_feats.join(pd.get_dummies("act_day_" + train["date_act"].dt.day.astype(str)))
    train_feats = train_feats.join(pd.get_dummies("act_month_" + train["date_act"].dt.month.astype(str)))
    train_feats = train_feats.join(pd.get_dummies("act_weekday_" + train["date_act"].dt.weekday.astype(str)))
    test_feats = test_feats.join(pd.get_dummies("act_day_" + test["date_act"].dt.day.astype(str)))
    test_feats = test_feats.join(pd.get_dummies("act_month_" + test["date_act"].dt.month.astype(str)))
    test_feats = test_feats.join(pd.get_dummies("act_weekday_" + test["date_act"].dt.weekday.astype(str)))

    # Seasonal features for people date, let's only do month and year because it seems to be a longer term thing
    train_feats = train_feats.join(pd.get_dummies("people_month_" + train["date"].dt.month.astype(str)))
    train_feats = train_feats.join(pd.get_dummies("people_year_" + train["date"].dt.year.astype(str)))
    test_feats = test_feats.join(pd.get_dummies("people_month_" + test["date"].dt.month.astype(str)))
    test_feats = test_feats.join(pd.get_dummies("people_year_" + test["date"].dt.year.astype(str)))

    # Drop those date columns now that they're not needed
    train = train.drop(["date", "date_act"], axis=1)
    test = test.drop(["date", "date_act"], axis=1)

    # Drop any columns with nans
    train = train.dropna(axis=1)
    test = test.dropna(axis=1)

    # Also remove group_1 since it's already used, and people_id by association. Also, outcome isn't a training feature
    outcomes = train["outcome"]
    train = train.drop(["group_1", "outcome", "people_id"], axis=1)
    test = test.drop(["group_1", "people_id"], axis=1)
    try:
        test = test.drop("outcome", axis=1)
    except:
        pass

    # Char_38 can go in as is since it's an ordinal feature, so long as we scale it to be between 0 and 1
    train_feats["char_38"] = train["char_38"]/100
    train = train.drop("char_38", axis=1)
    test_feats["char_38"] = test["char_38"]/100
    test = test.drop("char_38", axis=1)

    # Copy over the ready made booleans, one-hot the categorical with a reasonable number of features, drop as we go
    for column in train.columns.values:
        if train[column].dtype == bool:
            train_feats[column] = train[column].astype(int)
            train = train.drop(column, axis=1)
        elif len(set(train[column].tolist())) < 100:
            train_feats = train_feats.join(pd.get_dummies(column + "_" + train[column]))
            train = train.drop(column, axis=1)


    for column in test.columns.values:
        if test[column].dtype == bool:
            test_feats[column] = test[column].astype(int)
            test = test.drop(column, axis=1)
        elif len(set(test[column].tolist())) < 100:
            test_feats = test_feats.join(pd.get_dummies(column + "_" + test[column]))
            test = test.drop(column, axis=1)

    # Cross check for any columns that don't exist in both sides of the split
    for column in train_feats.columns.values:
        if column not in test_feats.columns.values:
            train_feats = train_feats.drop(column, axis=1)
    for column in test_feats.columns.values:
        if column not in train_feats.columns.values:
            test_feats = test_feats.drop(column, axis=1)

    # Let's get some PCA done on this giant data set, incrementally for my poor little RAM sticks
    decomp = IncrementalPCA(n_components=20)
    decomp.fit(train_feats, outcomes)

    # Prepare the training output
    train_comps = pd.DataFrame(decomp.transform(train_feats), index=train_feats.index)
    train_comps.columns = ["principle_component_" + str(i) for i in range(len(train_comps.columns.values))]

    # Prepare the testing output
    test_comps = pd.DataFrame(decomp.transform(test_feats), index=test_feats.index)
    test_comps.columns = ["principle_component_" + str(i) for i in range(len(test_comps.columns.values))]

    return train_comps, test_comps
