#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:38:55 2018

@author: user
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

# import the dataset
df=pd.read_csv('credit_data.csv')

# split the data into feature matrix and dependent variable
X=df[['income', 'age', 'loan']]
y=df['default']

# instantiate the logistic regression model 
model=LogisticRegression()

# predict prices
y_pred=cross_validation.cross_val_predict(model, X, y, cv=10)

# evaluate the model
accuracy=accuracy_score(y, y_pred)
