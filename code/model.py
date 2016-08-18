from utilities import simple_load
import pandas as pd
import numpy as np


def group_decision(train, test):
    # Exploit the leak revealed by Loiso and team to try and directly infer any labels that can be inferred
    # https://www.kaggle.com/c/predicting-red-hat-business-value/forums/t/22807/0-987-kernel-now-available-seems-like-leakage

    # Create some group filling columns for later use
    train["group_fillfw"] = train["group_1"]
    train["group_fillbw"] = train["group_1"]

    # Put the two data sets together and sort
    df = train.append(test)
    df = df.sort_values(by=["group_1", "date_act"])

    # Fill labels
    df["outcome_fillfw"] = df["outcome"].fillna(method="ffill")
    df["outcome_fillbw"] = df["outcome"].fillna(method="bfill")

    # Fill the groups
    df["group_fillfw"] = df["group_fillfw"].fillna(method="ffill")
    df["group_fillbw"] = df["group_fillbw"].fillna(method="bfill")

    # Use the filled labels if the labels were from the same group
    df["same_group"] = df["group_fillfw"] == df["group_fillbw"]
    df["interfill"] = (df["outcome_fillfw"]+df["outcome_fillbw"])/2
    df["outcome"] = df["outcome"].fillna(df["interfill"])
    df = df[df["same_group"] == True]

    # Paste outcomes to the original index
    test["outcome"] = df["outcome"]

    return test["outcome"]


def benchmark_model():

    # Load in the data set simply by merging together
    train, test = simple_load()

    # Try to just cluster together dates by each
    test["outcome"] = group_decision(train, test)

    # Write the inferred predictions to a template
    test.reset_index()[["activity_id", "outcome"]].to_csv("../output/starter_template.csv")

    # Fill any missing rows with the mean of the whole column
    test["outcome"] = test["outcome"].fillna(test["outcome"].mean())

    return test.reset_index()[["activity_id", "outcome"]]


def main():

    # Write a benchmark file to the submissions folder
    benchmark_model().to_csv("../output/benchmark_submission.csv", index=False)

if __name__ == "__main__":
    main()