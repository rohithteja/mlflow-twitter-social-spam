# MLflow pipeline of twitter spam detection problem

This repository holds the code for an MLflow project with Twitter spam dataset where the objective is to detect the spam users. In this project, SVM classifier is used to correctly predict the spam users. The hyperparameters "C" and "Kernel" can be tuned accordingly while performing the experiments.

Note: This MLflow project can be executed in conda virtual environment or by creating a docker container. The process is explained in the sections below:

## Using conda virtual environment

### Requirements:

Install MLflow from PyPI via `pip install mlflow`

MLflow requires `conda` to be on the `PATH` for the projects feature. 

### To run the project

Using command line interface

`git clone https://github.com/rohithteja/mlflow-twitter-social-spam.git`

`cd mlflow-twitter-social-spam`

`mlflow run conda-venv -P C=10 -P kernel="rbf"`

## Using docker container

### Requirements
Build the docker container using the repo (https://github.com/rohithteja/docker-img-socialspam)

### To run the project

`mlflow run docker-cont -P C=10 -P kernel="rbf"`



## MLflow ui can be accessed by 

`mlflow ui`


## Reference
https://github.com/mlflow/mlflow






