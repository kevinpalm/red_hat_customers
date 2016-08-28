import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def cluster_weight(train, test, predicts):

    # Fill in predictions and append together
    test["outcome"] = predicts["outcome"]
    df = train.append(test)

    # Calculate the days that the group has been ongoing
    df["date_act"] = pd.to_datetime(df["date_act"])
    grp = df[["group_1", "date_act"]].groupby("group_1", as_index=False)["date_act"].min()
    grp.columns = ["group_1", "group_min_date"]
    df = df.merge(grp, how="left", on="group_1")
    df["days_ongoing"] = df["group_min_date"] - df["date_act"]

    # Create a blank column for storing outputs
    df["group_cluster"] = None

    # Iterate through the groups
    for group in list(set(df["group_1"].tolist())):
        grpfeats = df.loc[df["group_1"]==group, ["days_ongoing", "outcome"]]

        # Create a blank dictionary for storing scores
        scores = {}

        # Run


        df.loc[df["group_1"]==group]

