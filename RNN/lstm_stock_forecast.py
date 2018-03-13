#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:55:23 2018

@author: user
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# load the datasets
prices_train=pd.read_csv('SP500_train.csv')
prices_test=pd.read_csv('SP500_test.csv')

# select the relevant columns
training_set=prices_train.iloc[:, 5:6].values
test_set=prices_test.iloc[:, 5:6].values

# normalize the training datatset
scaler=MinMaxScaler(feature_range=(0, 1))
scaled_training_set=scaler.fit_transform(training_set)

# prepare the datasets for forcasting
X_train=[]
y_train=[]

for i in range(40, 1258):
    X_train.append(scaled_training_set[i-40:i, 0])
    y_train.append(scaled_training_set[i, 0])
    
X_train=np.array(X_train)
y_train=np.array(y_train)

X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# build the LSTM model
model=Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# test the model
dataset_total=pd.concat((prices_train['adj_close'], prices_test['adj_close']), axis=0)
inputs=dataset_total[len(dataset_total)-len(prices_test)-40:].values
inputs=inputs.reshape(-1, 1)
inputs=scaler.transform(inputs)

X_test=[]

for i in range(40, len(prices_test)+40):
    X_test.append(inputs[i-40: i, 0])

X_test=np.array(X_test)
X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictions=model.predict(X_test)
predictions=scaler.inverse_transform(predictions)

# plot the results
plt.plot(test_set, color='blue', label='actual prices')
plt.plot(predictions, color='green', label='lstm predictions')
plt.title('SP predictions')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

# save and load the trained model for future predictions
from keras.models import load_model

model.save('mnist_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('mnist_model.h5')
