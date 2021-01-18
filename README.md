# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this project we want to compare HyperDrive and AutoML. The goal is to use HyperDrive to find the best hypeparameter for our trainig model and then compare it with AutoML and the model found by AutoML. 

The project is based on the UCI bank marketing data dataset and the _classification_ goal is to predict whether a user will subscribe a term deposit.

The dataset is composed of 21 columns split into bank data client:

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
* 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

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


The best model was the **VotingEnsemble** found by AutoML with an accuracy of **0.91612**.

## Scikit-learn Pipeline
As mentioned we have two approach: hyperparameter tuning and AutoML.

#### Hyperparameter tuning
In this part of the project we want to optimize our model using HyperDrive. 
HyperDrive is a tool in Azure ML that helps tuning hyperparameters.
The algorithm in the train.py is sklearn _LogisticRegression_ and the
 hyperparameters to optimize are 
* *C*: Inverse of regularization strength; must be a positive float.
 Like in support vector machines, smaller values specify stronger regularization.
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

Values chosen are:
* C: choice(0.001, 0.01, 0.1, 0.5, 1,  10, 100): this range has been chosen as C must be a positive number with default value = 1 
* max_iter: choice(20,40,60,80,100,1000)

**What are the benefits of the early stopping policy you chose?**

We use an early termination policy to ensure that poorly performing runs are stopped and so we can concentrate on good runs. 
The *BanditPolicy* has been chosen. The policy terminates any runs that doesn't satisfy the specific slack factor of the best performing run.
*Median stopping* is a policy based on running averages of primary metrics and stops all runs with primary metric values worse than the median of averages, while *Truncation selection* cancels a percentage (that must be set) of lowest performing runs at each evaluation interval.

Parameters chosen are:
* slack_factor = 0.1 -> It will evaluate every run and will stop a run if the best metric is less then 91% of the best performing
* evaluation_interval=1 -> will apply the policy every time the training script reports the primary metric

## AutoML
The second part of the project is to use AutoML to find a model that will be compared with previous. 
In the notebook we load the same CSV file using _TabularDatasetFactory_. 
Then data is cleaned with the _clean_data_ method imported from train.py.
The AutoMLConfig is configured using: 
* 'classification' as task 
* 'accuracy' as metric

AutoML creates a number of pipelines in parallel that try different algorithms and parameters for you. The service iterates through ML algorithms paired with feature selections, where each iteration produces a model with a training score.
Ensemble models are supported and enabled by default in AutML. Ensemble learning improves machine learning results and predictive performance by combining multiple models as opposed to using single models. 
The model was in fact an ensebmle model: **VotingEnsemble**. 
Ensemble models supported in AutoML are *voting* and *stacking*.
Voting: predicts based on the weighted average of predicted class probabilities (for classification tasks) or predicted regression targets (for regression tasks).

From the json file describing the fitted model I can see that algorythms in the model are: XGBoostClassifier, XGBoostClassifier, XGBoostClassifier, LightGBM, XGBoostClassifier, XGBoostClassifier, RandomForest.
From the same json file I could find the following hyperparameters associated:
* min_samples_split=0.2442105263157895
* min_weight_fraction_leaf=0.0
* n_estimators=10
* n_jobs=1
* oob_score=False
* random_state=None
* verbose=0
* warm_start=False


![Top 9 features](https://github.com/emanuelef1/Optimizing_a_Pipeline_in_Azure_EF/blob/master/images/all_models.png)

## Pipeline comparison
With the above configuration AutoML took a bit longer to run, but found the best model with an accuracy of **0.91612**,
 while accuracy for HyperDrive model was **0.91098**
 
AutoML found the best model that was an _ensemble_ model. The second best in the list was also 
an ensemble model, which shows us ensemble models perform better.
The Voting Ensemble combines predictions from multiple models and can find better performance
compared to single models.
Even though I wasn't able to discover what models have been used in the model.

AutoML was much easier to configure with less code to write and yet resulted to be more powerful, with AutoML 
is also very easy try out different algorithms compared to the single for the default approach.

Having a look at the top features for the model makes sense to me that duration, which shows the last contact duration, is the top feature as described above: if the duration is 0 so y = NO, and after the end of the call y is obviously known.
![Top 9 features](https://github.com/emanuelef1/Optimizing_a_Pipeline_in_Azure_EF/blob/master/images/top_9_features.png)

## Future work
I would like to make more runs to improve the hyperparameters and maybe try other algorithms for HyperDrive other than
LogisticRegression.
In particular I'd like to try out the Grid or Bayesan sampler starting form the results 
got from the results got with the Random sampler which should be 
our starting point. 
that sampling strategy will found a better accuracy. 
I'd also try to modify data removing the 'duration' column and using the duration as benchmark, as suggestend in the dataset description.

About AutoMl there not much to do, I guess. Doing more runs I've got different results (the latter the best actually) 
and maybe longer run would help finding better models.   

## Proof of cluster clean up
Cluster has been deleted in the notebook.
