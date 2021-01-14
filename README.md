# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this project we want to compare HyperDrive and AutoML. The goal is to use HyperDrive to find the best hypeparameter for our trainig model and then compare it with AutoML and the model found by AutoML. 

The project is based on the UCI bank marketing data dataset and the _classification_ goal is to predict whether a user will subscribe a term deposit.

The dataset is composed by 21 columns split into bank data client:

* 1 - age (numeric)
* 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
* 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
* 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
* 5 - default: has credit in default? (categorical: 'no','yes','unknown')
* 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
* 7 - loan: has personal loan? (categorical: 'no','yes','unknown')

and other attributes:
* 8 - contact: contact communication type (categorical: 'cellular','telephone')
* 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
* 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
* 11 - duration

other attributes:
* 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
* 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
* 14 - previous: number of contacts performed before this campaign and for this client (numeric)
* 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

social and economic context attributes
* 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
* 17 - cons.price.idx: consumer price index - monthly indicator (numeric)
* 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
* 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
* 20 - nr.employed: number of employees - quarterly indicator (numeric)

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best model was the **VotingEnsemble** found by AutoML with an accuracy of **0.91593**.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

As mentioned we have two approach: hyperparameter tuning and AutoML.

## Hyperparameter tuning
In this part of the architecture we want to optimize our model using HyperDrive. HyperDrive is a tool in Azure ML that helps tuning hyperparameters.
The algorithm in the train.py is sklearn _LogisticRegression_ and the hyperparameters to optimize are 
* *C*: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
* *max-iter*: Maximum number of iterations taken for the solvers to converge.


The architecture is composed by a train script and a notebook.
In the _train.py_ script the CSV file data is loaded into a Azure tabular Dataset from a URL, then data is cleaned and split into train and test sets.

In the notebook a HyperDriveConfig is created passing a parameter sampler, a early termination policy and an estimator.
In the configuration is also defined the metric to optimize, in this case we want to _maximize_ the _Accuracy_, and the number of run, which has been set to 20. 

Then the hypedrive run is submitted: what is does is run 20 (in our case) experiments with value randomly chosen by the Sampler. 

**What are the benefits of the parameter sampler you chose?**
The sampler chosen is the RandomParameterSampling where hyperparameter values are randomly selected from the defined search space. RandomParameterSampling is the first choice for an initial search before to refine the search space to improve results.
Both Grid sampling, which exhaustively search over the search space, and Bayesian sampling require enough budget. 
The Random sampling has been chosen has first method and also because of the VM timeout. 

C_param_range = [0.001,0.01,0.1,1,10,100]

Values chosen are:
* C: uniform(0.05, 1): this range has been chosen has C must be a positive number and 
* max_iter: choice(20,40,60,80,100,1000)

**What are the benefits of the early stopping policy you chose?**
We use an early termination policy to ensure that poorly performing runs are stopped and so we can concentrate on good runs. 
The *BanditPolicy* has been chosen. The policy terminates any runs that doesn't satisfy the specific slack factor of the best performing run.
*Median stopping* is a policy based on running averages of primary metrics and stops all runs with primary metric values worse than the median of averages, while *Truncation selection* cancels a percentage (that must be set) of lowest performing runs at each evaluation interval.

Parameters chosen are:
* slack_factor = 0.1 -> It will evaluate every run and will stop a run if the best metric is less then 91% of the best performing
* evaluation_interval=1 -> will apply the policy every time the training script reports the primary metric

![alt text](https://github.com/emanuelef1/Optimizing_a_Pipeline_in_Azure/blob/master/hyperdrive.png)

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
The second part of the project is to use AutoML to find a model. 
In the notebook we load the same CSV file using _TabularDatasetFactory _. 
Then data is cleaned importing the _clean_data method_ from train.py.
The AutoMLConfig is configured using: 
* 'classification' as task 
* 'accuracy' as metric

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
