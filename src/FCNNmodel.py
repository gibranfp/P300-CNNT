#!/usr/bin/env python3
# -*- coding: utf-8
#

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras import backend as K

def FCNN():
    
    eeg_input       = Input(shape = (1236,), name = 'EEG')
    dense           = Dense(2, activation = 'tanh',
                         kernel_initializer = 'glorot_uniform', name = 'Dense1')(eeg_input)
    flat            = Flatten()(dense)
    prediction      = Dense(1, activation = 'sigmoid', name = 'Dense2')(flat)
    

    return Model(inputs = eeg_input, outputs = prediction, name = 'FCNN')




