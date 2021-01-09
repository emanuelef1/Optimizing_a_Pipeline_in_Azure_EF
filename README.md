# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
This dataset is based on the UCI bank marketing data set and the goal is to predict whether a user will subscribe a term deposit.

The dataset is composed by 21 columns split into bank data client and other attributes.

We'll use HyperDrive to tune the hyperdrive parameters for the algorithm and compare to AutoML.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best model was... 

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
Firstly we start a compute cluster to run our exepriments.
We define a sampler, a bandit policy and and estimator. 
Then we run the experiment submissing the hyper drive configuration. 

The training script is 'train.py': in the script data is downloaded, cleaned and split.
The provided script has been modified to work correctly. In specific the declaration of method clean data has been moved and return x and y has been added. 

The algorithm in the train.py is LogisticRegression and the metric is Accuracy (with capital A).

**What are the benefits of the parameter sampler you chose?**
The sampler chosen is the RandomParameterSampling where hyperparameter values are randomly selected from the defined search space. 
Values chosen are:
C: uniform(0.05, 1)
max_iter: choice(20,40,60,80,100,1000)

**What are the benefits of the early stopping policy you chose?**
BanditPolicy has been chosen. The policy terminates any runs that doesn't satisfy the specific slack factor of the best performing run.
Parameters chosen are:
slack_factor = 0.1
evaluation_interval=1

It will evaluate every run and will stop a run if the best metric is less then 91% of the best performing.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
