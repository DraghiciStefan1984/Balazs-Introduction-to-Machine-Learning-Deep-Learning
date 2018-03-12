# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:46:07 2018

@author: Stefan Draghici
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

training_data=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], 'float32')
target_data=np.array([[0], [1], [1], [0]], 'float32')

model=Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=1000, verbose=2)

print(model.predict(training_data).round())