import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import numpy as np

def get_nn_model(input_size):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(input_size,)))
    model.add(Dense(units=1000, activation="relu",kernel_initializer="lecun_normal"))
    model.add(Dropout(0.2, input_shape=(1000,)))
    model.add(Dense(units=500, activation="relu",kernel_initializer="lecun_normal")) 
    model.add(Dense(units=250, activation='relu',kernel_initializer="lecun_normal"))
    model.add(Dense(units=30, activation='relu',kernel_initializer="lecun_normal"))
    model.add(Dense(1, activation='sigmoid',kernel_initializer="lecun_normal"))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    
    model.summary()
    return model