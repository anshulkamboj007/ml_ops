import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import metrics
import os
import mlflow


#load dataset
dataset=pd.read_csv("train.csv")
numerical_cols=dataset.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols=dataset.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')

#filling  col na

for col in categorical_cols:
    dataset[col].fillna(dataset[col].mode()[0],inplace=True)

for col in numerical_cols:
    dataset[col].fillna(dataset[col].median(),inplace=True)

#outliers

dataset[numerical_cols]=dataset[numerical_cols].apply(lambda x : x.clip(*x.quantile([0.05,0.95])))