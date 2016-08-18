from utilities import simple_load, group_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def group_decision(train, test):
    # Exploit the leak revealed by Loiso and team to try and directly infer any labels that can be inferred
    # https://www.kaggle.com/c/predicting-red-hat-business-value/forums/t/22807/0-987-kernel-now-available-seems-like-leakage

    # Create blank data frame for predictions of test
    predicts = pd.DataFrame()

    # Define an estimator to scale and then label each set of data
    estimator = Pipeline([("scale", MinMaxScaler()), ("regress", RandomForestRegressor())])

    # For each people group in the testing data...
    for group in test["group_1"].unique():
        try:

            # Get features for the subsection of interest
            subtrain_x, subtrain_y, subtest_x = group_split(group, train, test)

            # Train and join predictions
            estimator.fit(subtrain_x, subtrain_y)
            subtrain_x["outcome"] = subtrain_y
            subtest_x["outcome"] = estimator.predict(subtest_x)
            predicts = predicts.append(subtest_x)
        except:
            pass

    # Copy to the original index
    test["outcome"] = predicts["outcome"]

    return test["outcome"]


def benchmark_model():

    # Load in the data set simply by merging together
    train, test = simple_load()

    # Try to just cluster together dates by each
    test["outcome"] = group_decision(train, test)

    # Write the inferred predictions to a template
    test[["activity_id", "outcome"]].to_csv("../output/starter_template.csv")

    # Fill any missing rows with the mean of the whole column
    test["outcome"] = test["outcome"].fillna(test["outcome"].mean())

    return test[["activity_id", "outcome"]]


def main():

    # Write a benchmark file to the submissions folder
    benchmark_model().to_csv("../output/benchmark_submission.csv", index=False)

if __name__ == "__main__":
    main()