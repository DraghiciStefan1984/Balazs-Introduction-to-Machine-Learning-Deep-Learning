#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:53:29 2018

@author: user
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

# load the data and split into training and test sets
(X_train, y_train), (X_test, y_test)=mnist.load_data()

# convert the feature matrix according to tensorflow specifications
X_train=X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train=X_train.astype('float32')
X_test=X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test=X_test.astype('float32')

# normalize the values in the range [0, 1], as we are dealing with grey scale images
X_train/=255
X_test/=255

# transform the output labels so they are either 0 or 1, 
# for example: label 2 out of 10 will be [0,0,1,0,0,0,0,0,0,0]
y_train=np_utils.to_categorical(y_train, 10)
y_test=np_utils.to_categorical(y_test, 10)

#instantiate the neural network
model=Sequential()

# add the first convolutional layer with 32 filters of size 3 x 3,
# and the size of the input image will be 28 x 28 pixel with 1 channel 
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))

# add the relu activation function for the input layer
model.add(Activation('relu'))

# add batch normalization to normalize the activations from the previous layer
# and to maintain the mean activation close to 0 and standard deviation close to 1
model.add(BatchNormalization())

# add another convolutional layer of the same size and same activation function
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

# add a amx pool layer with the size 2 x 2 to deal with spatial invariance
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# add a third convolutional layer with 63 filters of size 3 x 3
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# add the flatten layer
model.add(Flatten())

# finally, add a fully connected feed forward network
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

# add regularization to prevent overfitting
model.add(Dropout(0.2))

# add the output layer with 10 output classes and the softmax activation function
model.add(Dense(10, activation='softmax'))

# compile the model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model to the training set
model.fit(X_train, y_train, batch_size=128, epochs=2, validation_data=(X_test, y_test), verbose=1)

# evaluate the model
score=model.evaluate(X_test, y_test)
print("Accuracy: ", score[1])

# save and load the trained model for future predictions
from keras.models import load_model

model.save('mnist_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('mnist_model.h5')