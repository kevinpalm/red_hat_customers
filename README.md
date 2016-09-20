## Introduction
This project contains code and a report pertaining to my work on the
August 1st - November 1st, 2016 Red Hat Kaggle Competition. In addition
to a submitted entry to the competition, I also hope to use my work here
as my Udacity Machine Learning Nanodegree capstone project. I've chosen
this specific project because it relates a lot to my current job, in
which I work as part of a business development department. For a
specific introduction to the competition and its goals, and for an
overview of my project, see the report file.

## Set Up
**Important**: If you want to generate a Kaggle submission using this
script, you should have at least 12 gigs of memory available for the
PCA step. Otherwise, you can still run the local tests without straining
your system. Or, you can edit line 232 of utilities.py so that the
IncrimentalPCA batch_size argument is something that your computer can
handle.

This project was written using Python 2.7, but with my best efforts to
make it Python 3 compatible. Required libraries are:

* [pandas](http://pandas.pydata.org/) and its dependencies
* [scikit-learn](http://scikit-learn.org/) and its dependencies

Also, in order to run this repository
[you'll need to download the competition data files](https://www.kaggle.com/c/predicting-red-hat-business-value/data),
and so you'll probably need to agree to the competition rules/terms.
Store the data files in a directory named "data" inside the repository
root directory, and you'll be all set up.

## Execution
The primary code is located in model.py, and to execute the script
you'll just need to execute model.py.

If you're only interested in running the local tests, make sure to
comment out the the extra lines in the main() function. The local tests
take about five minutes, whereas the full script to generate a kaggle
submission takes about two hours.

## Explaination of my Final Submission
On its own, this script scored 0.991392 AUC on the public leaderboard.
By using my cluster_weight postprocessing script on
jlowery's adaptation of loiso and Raddar's kernel scripts, then
averaging those results into my model, I was able to attain a score of
0.992279 AUC, which put me in the top 3 percent.

The postprocessing script essentially looks for clusters of datapoints -
with the datestamp as the only feature - for each group_1. Then it
adjusts the whole cluster towards the most extreme prediction in that
cluster, so long as the cluster doesn't straddle 0.5. In that case the
whole cluster is assumed as uncertain.

Because it takes a long time to create the map of those clusters,
I've included an already generated csv in this output directory of this
repo.