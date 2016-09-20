import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import Birch
from toolstuff import simple_load

def cluster_weight(train, test, cap=0.35, weight=1.0):

    # Try and retrieve the mapping if its already been created
    try:
        # Append together train and test
        df = train.append(test)

        # Read in the map file
        map = pd.read_csv("../output/group_cluster_map.csv").set_index("activity_id")

        # Join the map into the df
        df = df.join(map)


    except:
        print("Couldn't find the cluster map. Creating a new one. (Takes a long time...)")

        # Read in the whole dataset
        wholetrain, wholetest = simple_load()
        df = wholetrain.append(wholetest).reset_index()

        # Calculate the days that the group has been ongoing
        df["date_act"] = pd.to_datetime(df["date_act"])
        grp = df[["group_1", "date_act"]].groupby("group_1", as_index=False)["date_act"].min()
        grp.columns = ["group_1", "group_min_date"]
        df = df.merge(grp, how="left", on="group_1")
        df["days_ongoing"] = ((df["date_act"]-df["group_min_date"])/np.timedelta64(1, 'D')).astype(int)

        # Create a blank column for storing outputs
        df["group_cluster"] = None

        # Define clustering algorithm
        cluster = Pipeline([("scale", RobustScaler()), ("cluster", Birch(threshold=0.1, n_clusters=None))])

        # Iterate through the groups
        for group in list(set(df["group_1"].tolist())):
            grpfeats = df.loc[df["group_1"] == group, ["days_ongoing"]]
            df.loc[df["group_1"] == group, "group_cluster"] = cluster.fit_predict(grpfeats)

        # Write to a csv for examining the groups
        df[["activity_id", "group_cluster", "days_ongoing"]].to_csv("../output/group_cluster_map.csv")

        # Define the map columns
        map = df[["activity_id", "group_cluster", "days_ongoing"]].set_index("activity_id")

        # Append together train and test
        df = train.append(test)

        # Join the map into the df
        df = df.join(map)

    # Set any split dates to the average
    grp = df.groupby(["group_1", "date_act"], as_index=False)["outcome"].mean()
    df = df.drop("outcome", axis=1).reset_index().merge(grp, how="left", on=["group_1", "date_act"]).set_index(
        "activity_id")

    # Calculate the range of predictions per group per cluster
    grp = pd.DataFrame()
    grp["min_outcome"] = df.groupby(["group_1", "group_cluster"])["outcome"].min()
    grp["max_outcome"] = df.groupby(["group_1", "group_cluster"])["outcome"].max()

    # Calculate what's eligible to adjust which way
    grp["eligible_adjust_up"] = 0
    grp.loc[(grp["max_outcome"] - grp["min_outcome"] <= cap) & (grp["min_outcome"] > 0.5), "eligible_adjust_up"] = 1
    grp["eligible_adjust_down"] = 0
    grp.loc[(grp["max_outcome"] - grp["min_outcome"] <= cap) & (grp["max_outcome"] < 0.5), "eligible_adjust_down"] = 1

    # Calculate the adjustment to perform
    grp["adjustment"] = grp["max_outcome"]*grp["eligible_adjust_up"]+grp["min_outcome"]*grp["eligible_adjust_down"]

    # Do the adjustments
    grp = grp.reset_index()
    df = df.reset_index().merge(grp[["group_1", "group_cluster", "adjustment"]],
                  how="left",
                  on=["group_1", "group_cluster"]).set_index("activity_id")

    df["adjustment"] = df["adjustment"].replace(0.0, np.nan)
    df["adjusted"] = (df["adjustment"]*weight+df["outcome"])/(weight+1)
    df["adjusted"] = df["adjusted"].fillna(df["outcome"])
    df["outcome"] = df["adjusted"]

    # Apply adjustments
    test["outcome"] = df["outcome"]

    return test

def main():

    # Load the datasets
    train, test = simple_load()

    # Load the submission and join to test
    predicts = pd.read_csv("../output/kpalm_submission.csv").set_index("activity_id")
    test["outcome"] = predicts["outcome"]

    # Make adjustments
    test = cluster_weight(train, test, cap=0.35, weight=1.0)

    # Write to csv
    test.reset_index()[["activity_id", "outcome"]].to_csv("../output/adjusted_kpalm_submission.csv", index=False)


if __name__ == "__main__":
    main()