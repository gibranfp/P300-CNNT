1#!/usr/bin/env python3
# -*- coding: utf-8
#
# Montserrat Alvarado <amontserrat@gmail.com> / Gibran Fuentes-Pineda <gibranfp@unam.mx>
# DMAS-UAM / IIMAS-UNAM
# 2019
#

import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def BN3(Chans = 6, Samples = 206, Ns=16):
    
    input1       = Input(shape = (Samples, Chans))
    block1       = BatchNormalization(axis = 1)(input1)
    ##################################################################
    
    block1       = Conv1D(Ns, 1, stride=1, padding = 'same',
                            kernel_initializer='glorot_uniform',
                            activation='linear', use_bias = True)(block1)
    block1       = Conv1D(16, 20, stride=20, padding = 'same',
                            kernel_initializer='glorot_uniform',
                                       use_bias = True)(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('relu')(block1)    
    
    flatten      = Flatten(name = 'flatten')(block1)
    dense        = Dense(128, activation = 'tanh',
		   	 kernel_initializer='glorot_uniform')(flatten)
    block1       = Dropout(0.8)(dense)
    dense        = Dense(128, activation = 'tanh',
		   	 kernel_initializer='glorot_uniform')(flatten)
    block1       = Dropout(0.8)(dense)
    prediction   = Dense(1, activation = 'sigmoid')(block1)
    
    return Model(inputs=input1, outputs=prediction, name='BN3')


