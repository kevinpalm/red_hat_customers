from utilities import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from random import shuffle


def benchmark_model(train, test):

    # Try to just infer the correct dates using the data leak
    test["outcome"] = group_decision(train, test)

    # Fill any missing rows with the mean of the whole column
    test["outcome"] = test["outcome"].fillna(test["outcome"].mean())

    return test.reset_index()[["activity_id", "outcome"]]


def model(train, test):

    # Prep the features that are engineered to translate the data leak
    print("Starting the leak features extract...")
    train_x, train_y, test_x = extract_leak_features(train, test)

    # Get and join the regular feature principle components
    print("Starting the features extract and PCA...")
    train_feats, test_feats = prep_features(train, test, extra_outcomes=train_y)
    train_x = train_x.join(train_feats)
    test_x = test_x.join(test_feats)

    # Train an estimator
    print("Training the estimator...")
    estimator = GradientBoostingRegressor(learning_rate=0.3, n_estimators=100, max_depth=2)
    estimator.fit(train_x, train_y)

    # Predict and set the index back to the original testing index
    print("Making predictions...")
    test_x["outcome"] = estimator.predict(test_x)
    train_x["outcome"] = train_y
    df = train_x.append(test_x)
    test["outcome"] = df["outcome"]

    return test.reset_index()[["activity_id", "outcome"]]

def local_test(train, test):

    # Shuffle a list of people IDs for splitting, because the kaggle sets seem to be split on people_id
    people = list(set(train["people_id"].tolist()))
    shuffle(people)

    # Split the data into new training and testing sets
    test = train.loc[train["people_id"].isin(people[-2000:]), train.columns.values]
    train = train.loc[train["people_id"].isin(people[:2000]), train.columns.values]

    # Get predictions
    benchmark_predicts = benchmark_model(train, test.drop("outcome", axis=1))["outcome"]
    model_predicts = model(train, test.drop("outcome", axis=1))["outcome"]

    # Print the scores
    benchmark_score = roc_auc_score(test["outcome"].tolist(), benchmark_predicts.tolist())
    model_score = roc_auc_score(test["outcome"].tolist(), model_predicts.tolist())

    if benchmark_score < model_score:
        print("The model scored {0} AUC, and the benchmark scored {1} AUC. So this one is {2} AUC over the benchmark!"
              ).format(model_score, benchmark_score, model_score-benchmark_score)
    else:
        print("The model scored {0} AUC, and the benchmark scored {1} AUC. So this one is {2} AUC under the benchmark."
              ).format(model_score, benchmark_score, benchmark_score-model_score)

    return model_score-benchmark_score

def main():

    # Load in the data set by merging together
    train, test = simple_load()

    # Run 3 quick local tests
    scores = []
    for i in range(3):
        print("Local test {}...".format(i+1))
        scores.append(local_test(train, test))
    print "The average model score was {} over the benchmark.".format(sum(scores)/len(scores))

    # Write a benchmark file to the submissions folder
    print("Starting the benchmark model...")
    benchmark_model(train, test).to_csv("../output/benchmark_submission.csv", index=False)

    # Write model predictions file to the submissions folder
    print("Starting the main model...")
    model(train, test).to_csv("../output/kpalm_submission.csv", index=False)


if __name__ == "__main__":
    main()