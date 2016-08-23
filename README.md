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

Next you'll need to create a directory named "output" inside the
repository root directory, which is where model outputs will save to.

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
take about two minutes, whereas the full script to generate a kaggle
submission takes about 20 minutes.