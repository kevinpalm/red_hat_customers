from tools import *
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, make_scorer
from random import shuffle


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
    train_x["genre"] = train_x["left_outcome"].astype(str) + "-" + train_x["right_outcome"].astype(str)
    test_x["genre"] = test_x["left_outcome"].astype(str) + "-" + test_x["right_outcome"].astype(str)

    # Create a master list for appending predictions
    df = train_x.append(test_x)
    df["outcome"] = train_y
    df = df[["outcome"]]

    # Iterate through the genres so as to model separately
    for genre in list(set(train_x["genre"].tolist())):
        print("Modeling {} type data points...".format(genre))

        # Split the leak data by genre
        sub_train_x = train_x[train_x["genre"] == genre]
        sub_train_y = train_y[train_y.index.isin(sub_train_x.index)]
        sub_test_x = test_x[test_x["genre"] == genre]

        # Get feature data that pertains to the genre
        subtrain, subtest = subsplit_genre(train, test, sub_train_x[["genre"]], sub_test_x[["genre"]], sub_train_y)

        # Get and join the regular feature principle components
        train_feats, test_feats = prep_features(subtrain, subtest)
        sub_train_x = sub_train_x.join(train_feats)
        sub_test_x = sub_test_x.join(test_feats)

        # Drop the genres and the columns with nans
        sub_train_x = sub_train_x.drop("genre", axis=1).dropna(axis=1)
        sub_test_x = sub_test_x.drop("genre", axis=1).dropna(axis=1)

        # Train an estimator
        estimator = Pipeline([("select", VarianceThreshold()),
                              ("scale", RobustScaler()),
                              ("regress", AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=3,
                                                                                                 splitter="random"),
                                                            n_estimators=50000,
                                                            learning_rate=0.001))])
        estimator.fit(sub_train_x, sub_train_y)

        # Predict and set the index back to the original testing index
        sub_test_x["outcome"] = estimator.predict(sub_test_x)
        df["outcome"] = df["outcome"].fillna(sub_test_x["outcome"])

    test["outcome"] = df["outcome"]

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
    test = train.loc[train["people_id"].isin(people[-5000:]), train.columns.values]
    train = train.loc[train["people_id"].isin(people[:5000]), train.columns.values]

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

    return model_score-benchmark_score

def main():

    # Load in the data set by merging together
    train, test = simple_load()

    # Run 3 local tests... takes about 2 minutes
    scores = []
    for i in range(3):
        print("Local test {}...".format(i+1))
        scores.append(local_test(train, test))
    print("The average model score was {} AUC over the benchmark.".format(sum(scores)/len(scores)))

    # # Write a benchmark file to the submissions folder... takes about 30 seconds
    # print("Starting the benchmark model...")
    # benchmark_model(train, test).to_csv("../output/benchmark_submission.csv", index=False)

    # # Write model predictions file to the submissions folder... takes about 20 minutes
    # print("Starting the main model...")
    # model(train, test).to_csv("../output/kpalm_submission.csv", index=False)


if __name__ == "__main__":
    main()