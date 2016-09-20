from toolstuff import *
from postprocessing import cluster_weight
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from random import shuffle
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


def benchmark_model(train, test):

    """

    :param train: pandas.DataFrame
        The kaggle training dataset joined with the kaggle people dataset
    :param test:  pandas.DataFrame
        The kaggle testing dataset joined with the kaggle people dataset
    :return:
        test.reset_index()[["activity_id", "outcome"]] : pandas.DataFrame
            Formatted outcome predictions

    """

    # Try to just infer the correct dates using the data leak
    test["outcome"] = group_decision(train, test)

    # Fill any missing rows with the mean of the whole column
    test["outcome"] = test["outcome"].fillna(test["outcome"].mean())

    return test.reset_index()[["activity_id", "outcome"]]


def model(train, test):

    """

    :param train: pandas.DataFrame
        The kaggle training dataset joined with the kaggle people dataset
    :param test:  pandas.DataFrame
        The kaggle testing dataset joined with the kaggle people dataset
    :return:
        test.reset_index()[["activity_id", "outcome"]] : pandas.DataFrame
            Formatted outcome predictions

    """

    # Prep the features that are engineered to translate the data leak
    print("Starting the leak features extract...")
    train_x, train_y, test_x = extract_leak_features(train, test)

    # Divide up the dataset into types
    train_x["genre"] = train_x["left_outcome"].astype(str) + "-" +\
                       train_x["right_outcome"].astype(str)
    test_x["genre"] = test_x["left_outcome"].astype(str) + "-" +\
                      test_x["right_outcome"].astype(str)

    # Adjust the training features in terms of their difference from the group cluster means
    train_x, test_x = cluster_groups_delta(train.append(test)["group_1"], train_x, test_x)

    # Create a master list for appending predictions
    df = train_x.append(test_x)
    df["outcome"] = train_y
    df = df[["outcome"]]
    df["prediction"] = None

    # Iterate through the genres so as to model separately
    for genre in list(set(train_x["genre"].tolist())):
        print("Modeling {} type data points...".format(genre))

        # Split the leak data by genre
        sub_train_x = train_x[train_x["genre"] == genre]
        sub_test_x = test_x[test_x["genre"] == genre]
        sub_train_y = train_y[train_y.index.isin(sub_train_x.index)]

        # Check if there are at least two label types in the training examples
        if len(set(sub_train_y.tolist())) > 1:

            # Drop the columns with nans
            sub_train_x = sub_train_x.dropna(axis=1)
            sub_test_x = sub_test_x.dropna(axis=1)

            # Cross check for any columns that don't exist in both sides of the split
            for column in sub_train_x.columns.values:
                if column not in sub_test_x.columns.values:
                    sub_train_x = sub_train_x.drop(column, axis=1)
            for column in sub_test_x.columns.values:
                if column not in sub_train_x.columns.values:
                    sub_test_x = sub_test_x.drop(column, axis=1)

            # Get feature data that pertains to the genre
            subtrain, subtest = subsplit_genre(train, test, sub_train_x[["genre"]], sub_test_x[["genre"]], sub_train_y)

            # Get and join the regular feature components and selected features
            train_feats, test_feats = prep_features(subtrain, subtest)
            sub_train_x = sub_train_x.join(train_feats)
            sub_test_x = sub_test_x.join(test_feats)

            # Drop the genres
            sub_train_x = sub_train_x.drop("genre", axis=1)
            sub_test_x = sub_test_x.drop("genre", axis=1)

            # Define the default estimator
            transformer = Pipeline([("select", VarianceThreshold()),
                                    ("scale", RobustScaler()),
                                    ("decomp", PLSRegression(n_components=7))])
            estimator = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=3, splitter="random"),
                                          n_estimators=4000,
                                          learning_rate=0.01)

            try:
                sub_train_x = pd.DataFrame(transformer.fit_transform(sub_train_x, sub_train_y),
                                           index=sub_train_x.index)
                sub_train_x.columns = ["comp_" + str(i) for i in range(len(sub_train_x.columns.values))]
                sub_test_x = pd.DataFrame(transformer.transform(sub_test_x), index=sub_test_x.index)
                sub_test_x.columns = ["comp_" + str(i) for i in range(len(sub_test_x.columns.values))]

            except:
                sub_train_x = unpack_values(transformer.fit_transform(sub_train_x, sub_train_y))
                sub_train_x["activity_id"] = subtrain.index
                sub_train_x = sub_train_x.set_index("activity_id")
                sub_train_x.columns = ["comp_" + str(i) for i in range(len(sub_train_x.columns.values))]
                sub_test_x = unpack_values(transformer.transform(sub_test_x))
                sub_test_x["activity_id"] = subtest.index
                sub_test_x = sub_test_x.set_index("activity_id")
                sub_test_x.columns = ["comp_" + str(i) for i in range(len(sub_test_x.columns.values))]

            # Cross check for any columns that don't exist in both sides of the split
            for column in sub_train_x.columns.values:
                if column not in sub_test_x.columns.values:
                    sub_train_x = sub_train_x.drop(column, axis=1)
            for column in sub_test_x.columns.values:
                if column not in sub_train_x.columns.values:
                    sub_test_x = sub_test_x.drop(column, axis=1)

            # try:
            #     print("Plotting first two components as examples...")
            #     plot = sub_train_x.copy()
            #     plot["outcome"] = sub_train_y
            #     ax = plot[plot["outcome"] == 1].plot(kind="scatter", x="comp_0", y="comp_1", color="Orange", label="Type 1",
            #                                          alpha=0.2)
            #     plot[plot["outcome"] == 0].plot(kind="scatter", x="comp_0", y="comp_1", color="Blue", label="Type 0",
            #                                     ax=ax, alpha=0.2)
            #     plt.show()
            #     plt.clf()
            # except:
            #     print("Nothing to plot.")

            # Train an estimator
            estimator.fit(sub_train_x, sub_train_y)

            # Predict and set the index back to the original testing index
            sub_test_x["outcome"] = estimator.predict(sub_test_x)
            sub_train_x["prediction"] = estimator.predict(sub_train_x)

        else:
            sub_test_x["outcome"] = sub_train_y.mean()
            sub_train_x["prediction"] = sub_train_y.mean()

        df["outcome"] = df["outcome"].fillna(sub_test_x["outcome"])
        df["prediction"] = df["prediction"].fillna(sub_train_x["prediction"])

    # Make post adjustments
    print("Starting postprocessing adjustments...")
    df = same_extremes(df, train, test)
    test["outcome"] = df["outcome"]
    train["prediction"] = df["prediction"]
    test = cluster_weight(train, test, cap=0.35, weight=1.0)

    # Report the training score
    print("\n****\nThe model scores {} when used back on the "\
          "original training set".format(roc_auc_score(train["outcome"], train["prediction"])))

    return test.reset_index()[["activity_id", "outcome"]]


def local_test(train, test):

    """

    :param train: pandas.DataFrame
        The kaggle training dataset joined with the kaggle people dataset
    :param test:  pandas.DataFrame
        The kaggle testing dataset joined with the kaggle people dataset
    :return:
        model_score-benchmark_score : float
            Difference between the model score and the benchmark score

    """

    # Shuffle a list of people IDs for splitting, because the kaggle sets seem to be split on people_id
    people = list(set(train["people_id"].tolist()))
    shuffle(people)

    # Split the data into new training and testing sets
    test = train.loc[train["people_id"].isin(people[-10000:]), train.columns.values]
    train = train.loc[train["people_id"].isin(people[:10000]), train.columns.values]

    # Get predictions
    benchmark_predicts = benchmark_model(train, test.drop("outcome", axis=1))["outcome"]
    model_predicts = model(train, test.drop("outcome", axis=1))["outcome"]

    # Print the scores
    benchmark_score = roc_auc_score(test["outcome"].tolist(), benchmark_predicts.tolist())
    model_score = roc_auc_score(test["outcome"].tolist(), model_predicts.tolist())

    if benchmark_score < model_score:
        print("The model scored {0} AUC, and the benchmark scored {1} AUC. So this one is {2} AUC over the benchmark!"
              .format(model_score, benchmark_score, model_score-benchmark_score))
    else:
        print("The model scored {0} AUC, and the benchmark scored {1} AUC. So this one is {2} AUC under the benchmark."
              .format(model_score, benchmark_score, benchmark_score-model_score))

    print("\n****\n")

    return model_score-benchmark_score

def main():

    # Load in the data set by merging together
    train, test = simple_load()

    # Run 3 local tests... takes about 2 minutes
    scores = []
    for i in range(5):
        print("Local test {}...".format(i+1))
        scores.append(local_test(train, test))
    print("The average model score was {} AUC over the benchmark.".format(sum(scores)/len(scores)))

    # Write a benchmark file to the submissions folder... takes about 30 seconds
    # print("Starting the benchmark model...")
    # benchmark_model(train, test).to_csv("../output/benchmark_submission.csv", index=False)

    # # Write model predictions file to the submissions folder... takes about 20 minutes
    # print("Starting the main model...")
    # model(train, test).to_csv("../output/kpalm_submission.csv", index=False)


if __name__ == "__main__":
    main()