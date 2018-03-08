#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 20:51:44 2018

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math

# import the dataset
df=pd.read_csv('house_prices.csv')

# predict the house price based on the house size
house_size=df['sqft_living']
house_price=df['price']

# convert the vectors to np arrays
X=np.array(house_size).reshape(-1, 1)
y=np.array(house_price).reshape(-1, 1)

# split the model into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# fit and train the linear regression model 
model=LinearRegression()
model.fit(X_train, y_train)

# predict prices
y_pred=model.predict(X_test)

# calculate the mean squared error and r squared
mse=math.sqrt(mean_squared_error(y_test, y_pred))
r_squared=model.score(X_test, y_pred)

# get the coeficients
b0=model.coef_[0]
b1=model.intercept_[0]

# visualize the train results
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Train Results')
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.show()

# visualize the test results
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Test Results')
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.show()

# predict new prices
preds=model.predict([[5530], [3200], [6588], [1234], [9087]])