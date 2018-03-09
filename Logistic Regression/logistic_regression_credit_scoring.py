#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:38:55 2018

@author: user
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# import the dataset
df=pd.read_csv('credit_data.csv')

# split the data into feature matrix and dependent variable
X=df[['income', 'age', 'loan']]
y=df['default']

# split the model into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# fit and train the logistic regression model 
model=LogisticRegression()
model.fit(X_train, y_train)

# predict prices
y_pred=model.predict(X_test)

# evaluate the model
cm=confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)
