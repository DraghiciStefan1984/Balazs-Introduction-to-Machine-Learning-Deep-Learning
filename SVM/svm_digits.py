#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 18:14:24 2018

@author: user
"""

from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets

df=datasets.load_digits()
images_and_labels=list(zip(df['images'], df['target']))

n_samples=len(df['images'])
data=df['images'].reshape((n_samples, -1))

model=svm.SVC(gamma=0.001)

trainTestSplit=int(n_samples*0.75)
model.fit(data[:trainTestSplit], df['target'][:trainTestSplit])

expected=df['target'][trainTestSplit:]
predicted=model.predict(data[trainTestSplit:])

cm=confusion_matrix(expected, predicted)
accuracy=accuracy_score(expected, predicted)