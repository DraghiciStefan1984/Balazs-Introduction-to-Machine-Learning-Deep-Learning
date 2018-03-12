# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:46:07 2018

@author: Stefan Draghici
"""

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


dataset=load_iris()
X=dataset['data']
y=dataset['target'].reshape(-1, 1)

encoder=OneHotEncoder()

y=encoder.fit_transform(y)
y=y.toarray()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

model=Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(3, activation='softmax'))

optimizer=Adam(lr=0.005)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, batch_size=20, verbose=2)

results=model.evaluate(X_test, y_test)
print(results)
print("Accuracy is: ", results[1])