#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:54:13 2018

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn import datasets

X, y=datasets.make_moons(n_samples=1500, noise=0.05)

x1=X[:, 0]
x2=X[:, 1]

plt.scatter(x1, x2, s=5)
plt.show()

dbscan=DBSCAN(eps=0.1)
dbscan.fit(X)
dbscan_y_pred=dbscan.labels_.astype(np.int)

colors=np.array(['#ff0000', '#00ff00'])

plt.scatter(x1, x2, s=5, color=colors[dbscan_y_pred])
plt.show()

kmeans=KMeans(n_clusters=2)
kmeans.fit(X)
kmeans_y_pred=kmeans.labels_.astype(np.int)

plt.scatter(x1, x2, s=5, color=colors[kmeans_y_pred])
plt.show()