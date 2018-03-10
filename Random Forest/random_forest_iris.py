#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 21:35:25 2018

@author: user
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets

df=datasets.load_iris()

X=df['data']
y=df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model=RandomForestClassifier(max_features='sqrt')

param_grid={'n_estimators':[10, 100, 200, 500, 1000],
            'max_depth':[1, 5, 10, 15, 20],
            'min_samples_leaf':[1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]}


grid_search=GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

optimal_estimators=grid_search.best_params_.get('n_estimators')
optimal_depth=grid_search.best_params_.get('max_depth')
optimal_leaf=grid_search.best_params_.get('min_samples_leaf')

best_model=RandomForestClassifier(n_estimators=optimal_estimators, max_depth=optimal_depth, min_samples_leaf=optimal_leaf)
kfold=KFold(n_splits=10, random_state=123)

predictions=cross_val_score(best_model, X_test, y_test, cv=kfold.n_splits)

cm=confusion_matrix(y_test, predictions)
accuracy=accuracy_score(y_test, predictions)

