#!/usr/bin/env python3
# -*- coding: utf-8
#

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras import backend as K

def FCNN(Chans = 6, Samples = 206):
    
    input1       = Input(shape = (1,Chans,Samples))
    dense        = Dense(units=2, activation = 'tanh',
                         kernel_initializer='glorot_uniform', name='densa1')(input1)
    flat         = Flatten()(dense)
    prediction   = Dense(units=2, activation = 'sigmoid',name='densa2')(flat)
    

    return Model(inputs=input1, outputs=prediction, name='FCNN')




