# MLflow pipeline of twitter spam detection problem

This repository holds the code for an MLflow project with Twitter spam dataset where the objective is to detect the spam users. In this project, SVM classifier is used to correctly predict the spam users. The hyperparameters "C" and "Kernel" can be tuned accordingly while performing the experiments.

## Requirements:

Install MLflow from PyPI via `pip install mlflow`

MLflow requires `conda` to be on the `PATH` for the projects feature. 

## To run the project

Using command line interface

`mlflow run socialspam -P C=10 -P kernel="rbf"`

MLflow ui

`mlflow ui`

## Reference
https://github.com/mlflow/mlflow






