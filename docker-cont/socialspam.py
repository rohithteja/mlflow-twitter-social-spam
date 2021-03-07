import os
import warnings
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Import mlflow
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	np.random.seed(40)

	# read data and merge the train and test with features
	coded_id = pd.read_csv("data/users/coded_ids.csv")
	features = pd.read_csv("data/users_features/features.csv")
	users = pd.merge(features, coded_id, on='user_id', how='left')

	train = pd.read_csv("data/users/coded_ids_labels_train.csv")
	test = pd.read_csv("data/users/coded_ids_labels_test.csv")
	train = pd.merge(users, train, on='coded_id', how='right')
	test = pd.merge(users, test, on='coded_id', how='right')

	#cleaning train dataset

	#remove columns with na values
	nacolumns = train.columns[train.isna().any()].tolist()
	train.drop(columns = nacolumns,inplace=True)
	train.dropna(axis=0,inplace=True)

	#remove unwanted category columns like time and IDs
	unwanted = ["default_profile","default_profile_image","avg_intertweet_times","date_newest_tweet","lang","min_intertweet_times","std_nb_symbols_per_tweet","std_nb_symbols_per_word_in_the_tweet","date_oldest_tweet","max_intertweet_times","max_nb_symbols_per_tweet","max_nb_symbols_per_word_in_the_tweet","std_intertweet_times","user_id","coded_id"]
	train.drop(columns=unwanted,inplace=True)

	#remove columns like 
	allzero_cols = list(train.loc[:,(train==0).all()].columns)
	train.drop(columns=allzero_cols,inplace=True)

	#cleaning test dataset

	#remove columns with na values
	nacolumns = test.columns[test.isna().any()].tolist()
	test.drop(columns = nacolumns,inplace=True)
	test.dropna(axis=0,inplace=True)

	#remove unwanted category columns like time and IDs
	unwanted = ["default_profile","default_profile_image","avg_intertweet_times","date_newest_tweet","lang","min_intertweet_times","std_nb_symbols_per_tweet","std_nb_symbols_per_word_in_the_tweet","date_oldest_tweet","max_intertweet_times","max_nb_symbols_per_tweet","max_nb_symbols_per_word_in_the_tweet","std_intertweet_times","user_id","coded_id"]
	test.drop(columns=unwanted,inplace=True)

	#remove columns like 
	allzero_cols = list(test.loc[:,(test==0).all()].columns)
	test.drop(columns=allzero_cols,inplace=True)

	#split data into x features and labels
	x = train.iloc[:,:124]

	#normalizing the numerical features
	scale = StandardScaler()
	x = scale.fit_transform(x)
	test = scale.fit_transform(test)

	y = train.label

	#train test (validation) split
	x_train, x_test, y_train, y_test = train_test_split(x,y ,stratify=y, test_size=0.15,random_state=1)

	C = int(sys.argv[1]) if len(sys.argv) > 1 else 10
	kernel = str(sys.argv[2]) if len(sys.argv) > 2 else "poly"
	
	#svc test
	model = SVC(C=C,kernel=kernel)
	model.fit(x_train, y_train)
	y_pred_train = model.predict(x_train)
	y_pred_test = model.predict(x_test)
	train_acc = accuracy_score(y_train, y_pred_train)
	test_acc = accuracy_score(y_test, y_pred_test)
	test_f1 = f1_score(y_test, y_pred_test)

	print('SVC train accuracy score: {0:0.4f}'. format(train_acc))
	print('SVC test accuracy score: {0:0.4f}'. format(test_acc))
	print('SVC test F1 score: {0:0.4f}'. format(test_f1))

	# Log mlflow attributes for mlflow UI
	mlflow.log_param("C", C)
	mlflow.log_param("kernel", kernel)
	mlflow.log_metric("train_acc", train_acc)
	mlflow.log_metric("test_acc", test_acc)
	mlflow.log_metric("test_f1", test_f1)
	mlflow.sklearn.log_model(model, "model")