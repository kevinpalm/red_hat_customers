# Capstone Project
## Machine Learning Engineer Nanodegree
Kevin Palm  
August, 2016

## I. Definition

### Project Overview
The official project overview can be found on the
[competition description page](https://www.kaggle.com/c/predicting-red-hat-business-value).
The following overview is just my own paraphrasing and
interpretation of the competition goals/context.

Red Hat is a software company which specializes in open-source,
linux-based enterprise database solutions. They're interested in
improving their models which classify potential clients as future
clients, which is understandable because once that a company knows who
their serious prospects are they can devote extra resources to winning
them over.

They've just released three anonymized data sets for Kagglers to
use as they compete on creating that improved model.

The first data set is a one pertaining to the people in the data set -
potential and past clients - and what kind of characteristics those
people have. All of the characteristics are anonymized, in the sense
that the column headers are named things like "char_1" and "char_2",
rather than "Gender" or "Location". Most of the characteristics are
booleans. There is also an unlabeled date column and a feature called
"group_1". The people data set can be joined to the other two files
using a unique identifier field they provide.

The second data set is the competition training data, which similarly
contains a date column, a feature called "activity category", and 10
anonymized characteristic columns - this time presumably pertaining to
the characteristics of each activity. In addition, there is a final
column called "outcome", which has two potential values of 0 or 1, and
which is the feature which the competition has us attempting to predict.

The third data set is the testing data, which is exactly the same format
as the training data except that it lacks the "outcome" column.

The problem domain of this challenge definitely includes supervised
machine learning - specifically, the final model will be a classifier.
Also, I think this competition will be hugely a problem of exploratory
data analysis and observation. The anonymized features introduce a lot
of challenge in the sense that they remove most elements of business
intuition and make the contestants almost totally relent on raw data
analysis. But at the same time, that reliance on data analysis can
provide the benefit or eliminating researcher bias, in the sense that I
might give each feature a more thorough look into its applications than
if I were to come in with preconceptions about each feature's
usefulness.

I think this project has special application to business development and
marketing departments for companies which tailor to enterprises. The
framing of the challenge matches the overall theme of how many
businesses are attempting to automate and make data driven their
marketing and sales qualified leads.

### Problem Statement

Ultimately, the goal of this project is to create a list of true/false
predictions which append to the testing data set. The competition
guidelines don't specifically explain what it is that competitors are
predicting, but I would guess the general gist of it is something like
"Will this prospect become a client in the next thirty days?" So, given
that the training/testing data actually represents activities, I think a
good problem statement for this project would be this:

**As potential customers interact with Red Hat in the future, some of
their activities will be of interest and some will not, and we want
to know which so Red Hat can better use their selling resources.**

My expected tasks towards a solution to this problem statement are:

1. **Exploratory data analysis and data joining** - there's going to be
a lot of EDA required for this project. Understanding how that these
data sets were created, split up, and set up will be critical to
creating the right model in the end. I don't have the history or any
inside knowledge about the quirks of the data, and I'll need to know as
much as I can to create the right model. It will be an investigation.
During this phase, I'll need to join the people data set to the other
data sets.
2. **Feature preparation** - I'll format the data in an appropriate
manner for my final model. Exactly how I go about this will be hugely
reliant on the EDA step, as how I condition features and which algorithm
I intend to use will be dependant on what I've learned so far.
3. **Early Modelling** - I'll create a model that outputs predictions,
using a subset of the training set so that I can use classification
metrics on the leftover training data. I'll use the metric scores to
tune my model.
4. **Model** - I'll apply my model testing data which outputs a
predictions file in the correct format for submission to kaggle.com,
and submit the entry.
5. **Repeat** - I'll go back, learn more, and improve.

### Metrics

This Kaggle competition is scored on area under receiver operating
characteristic curve (AUC). AUC is a bit less intuitive of a
classification metric, but in the words of the top reply to
[this excellent blog post on the subject](http://fastml.com/what-you-wanted-to-know-about-auc/):

> Pick a random negative and a random positive example; The AUC gives
> you the probability that your classifier assigns a higher score to the
> positive example (ie, ranks the positive higher than the negative).

-- <cite>Peter Prettenhofer, also the same explanation is cited on
Wikipedia from "Fawcett, Tom (2006); An introduction to ROC analysis,
Pattern Recognition Letters, 27, 861–874"</cite>


In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
