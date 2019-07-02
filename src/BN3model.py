#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# DMAS-UAM / IIMAS-UNAM
# 2019
#
'''
Keras implementation of Liu's BN^3 architecture (https://www.sciencedirect.com/science/article/pii/S0925231217314601).
Requires 'image_data_format' to be set as 'channels_first' in keras.json config file. 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def BN3(Samples = 206, Chans = 6):
    eeg_input   = Input(shape = (Samples, Chans))
    bn_input    = BatchNormalization()(eeg_input)
    block1       = Conv1D(16, 1, strides = 1, padding = 'valid', use_bias = True,
                          data_format = 'channels_last', bias_initializer = 'glorot_uniform', kernel_initializer = 'glorot_uniform')(bn_input)
    block2       = Conv1D(16, 20, strides = 20, padding = 'same', use_bias = True,
                          data_format = 'channels_last', bias_initializer = 'glorot_uniform', kernel_initializer = 'glorot_uniform')(block1)
    block2      = BatchNormalization()(block2)
    block2       = Activation('relu')(block2)
    
    flatten      = Flatten(name = 'flatten')(block2)
    dense1       = Dense(128, activation = 'tanh', bias_initializer = 'glorot_uniform', kernel_initializer = 'glorot_uniform')(flatten)
    dense1       = Dropout(0.8)(dense1)
    dense2       = Dense(128, activation = 'tanh', bias_initializer = 'glorot_uniform', kernel_initializer = 'glorot_uniform')(dense1)
    dense2       = Dropout(0.8)(dense2)
    prediction   = Dense(1, activation = 'sigmoid', bias_initializer = 'glorot_uniform', kernel_initializer = 'glorot_uniform')(dense2)
    
    return Model(inputs = eeg_input, outputs = prediction, name = 'BN3')
