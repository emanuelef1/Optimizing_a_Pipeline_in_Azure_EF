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

The best model was the **VotingEnsemble** with an accuracy of **0.91557**.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

Firstly we start a compute cluster to run our exepriments.
We define a sampler, a policy and and estimator. 
Then we run the experiment submissing the hyper drive configuration. 

The training script is 'train.py': in the script data is downloaded, cleaned and split.
The provided script has been modified to work correctly. In details the declaration of method clean_data has been moved at the beginning of the file and return x and y has been added. 

The algorithm in the train.py is LogisticRegression and the metric is Accuracy (with capital A).

**What are the benefits of the parameter sampler you chose?**
The sampler chosen is the RandomParameterSampling where hyperparameter values are randomly selected from the defined search space. 

Values chosen are:
* C: uniform(0.05, 1)
* max_iter: choice(20,40,60,80,100,1000)

**What are the benefits of the early stopping policy you chose?**
BanditPolicy has been chosen. The policy terminates any runs that doesn't satisfy the specific slack factor of the best performing run.

Parameters chosen are:
* slack_factor = 0.1
* evaluation_interval=1

It will evaluate every run and will stop a run if the best metric is less then 91% of the best performing.
![alt text](https://github.com/emanuelef1/Optimizing_a_Pipeline_in_Azure/blob/master/hyperdrive.png)

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
AutoML pipeline is fed by the same dataset, then data is cleaned imoprting the clean_data method from train.py.
Then 'classification' is used as task and 'accuracy' as metric. 
AutoML runs more iterations to find the best performing model, which was VotingEnsemble.
![alt text](https://github.com/emanuelef1/Optimizing_a_Pipeline_in_Azure/blob/master/autoML.png)

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? 
In architecture? If there was a difference, why do you think there was one?**

WIth the above configuration AutoML took a bit longer to run, but found the best model wich an accuracy of **0.91557**, while accuracy for Hyperdrive model was **0.910336**
AutoML was much easier to configure with less code to write and yet resulted to be more powerful, with AutoML is also very easy try out different algorithms compared to the single one in hyperdrive.

## Future work
I would like to make more run to improve the hyperdrive model and try other algorithms for hyperdrive.

## Proof of cluster clean up
Cluster has been deleted in the notebook.
