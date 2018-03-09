# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:06:58 2018

@author: Stefan Draghici
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing

# import the dataset
df=pd.read_csv('credit_data.csv')

# split the data into feature matrix and dependent variable
X=df[['income', 'age', 'loan']]
y=df['default']

# normalize the data
X=preprocessing.MinMaxScaler().fit_transform(X)

# split the model into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# fit and train the KNN model 
model=KNeighborsClassifier(n_neighbors=28)
model.fit(X_train, y_train)

# make predictions
y_pred=model.predict(X_test)

# evaluate the model
cm=confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)

# find the optimal number of neighbors
cross_validation_scores=[]

for k in range(1, 101):
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    cross_validation_scores.append(scores.mean())
    
print("optimal k: ", np.argmax(cross_validation_scores))

cm=confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)