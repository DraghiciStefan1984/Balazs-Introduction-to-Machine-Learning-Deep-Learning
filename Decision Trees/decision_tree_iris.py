#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 21:35:25 2018

@author: user
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import cross_validation

df=pd.read_csv('iris_data.csv')

X=df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y=df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model=DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
predictions=model.predict(X_test)

cm=confusion_matrix(y_test, predictions)
accuracy=accuracy_score(y_test, predictions)

predicted=cross_validation.cross_val_predict(model, X, y, cv=10)
cross_val_accuracy=accuracy_score(y, predicted)