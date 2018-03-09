#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 18:14:24 2018

@author: user
"""

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets

df=datasets.load_iris()

X=df['data']
y=df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model=svm.SVC()
model.fit(X_train, y_train)
predictions=model.predict(X_test)

cm=confusion_matrix(y_test, predictions)
accuracy=accuracy_score(y_test, predictions)