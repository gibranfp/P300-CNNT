#!/usr/bin/env python3
# -*- coding: utf-8
#
# Montserrat Alvarado <amontserrat@gmail.com> / Gibran Fuentes-Pineda <gibranfp@unam.mx>
# DMAS-UAM / IIMAS-UNAM
# 2019
#
'''
 Keras implementation of Liu's BN^3 architecture (https://www.sciencedirect.com/science/article/pii/S0925231217314601). Requires 'image_data_format' to be set as 'channels_first' in keras.json config file. 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def BN3(Chans = 6, Samples = 206):
    input1       = Input(shape = (Samples, Chans))
    block1       = BatchNormalization()(block1)
    block1       = Conv1D(16, 1, stride = 1, padding = 'same', use_bias = True,
                          data_format = 'channels_last')(block1)
    block1       = Activation('relu')(block1)
    block2       = Conv1D(10, (20,1), stride=20, padding = 'same', use_bias = True,
                          data_format = 'channels_last')(block1)
    block2       = Activation('relu')(block2)    
    block2       = BatchNormalization()(block2)
    
    flatten      = Flatten(name = 'flatten')(block2)
    dense        = Dense(128, activation = 'tanh')(flatten)
    block1       = Dropout(0.8)(dense)
    dense        = Dense(128, activation = 'tanh')(block1)
    block1       = Dropout(0.8)(dense)
    prediction   = Dense(1, activation = 'sigmoid')(block1)
    
    return Model(inputs = input1, outputs = prediction, name = 'BN3')


