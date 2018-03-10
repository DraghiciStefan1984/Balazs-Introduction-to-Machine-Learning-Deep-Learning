#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:04:59 2018

@author: user
"""

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

df=pd.read_csv('iris_data.csv')

X=df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y=df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model=AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=123)
model.fit(X_train, y_train)
predictions=model.predict(X_test)

cm=confusion_matrix(y_test, predictions)
accuracy=accuracy_score(y_test, predictions)