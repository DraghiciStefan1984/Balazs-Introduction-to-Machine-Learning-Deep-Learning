#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:21:41 2018

@author: user
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

df=datasets.load_digits()
X=df['data']
y=df['target']

model=PCA(n_components=10)
X_pca=model.fit_transform(X)

colors=['back', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

for i in range(len(colors)-1):
    px=X_pca[:, 0][y==i]
    py=X_pca[:, 1][y==i]
    plt.scatter(px, py, c=colors[i], alpha=0.8)
    
plt.show()