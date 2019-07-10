#!/usr/bin/env python3
# -*- coding: utf-8
#
# Montserrat Alvarado <amontserrat@gmail.com> / Gibran Fuentes-Pineda <gibranfp@unam.mx>
# DMAS-UAM / IIMAS-UNAM
# 2019
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def streg(a):
    return 0.01 * K.sum(K.square(a[:, 1:, :] - a[:, :-1, :]))

def CNNR(Chans = 6, Samples = 206):
    eeg_input   = Input(shape = (Samples, Chans))

    block1       = Conv1D(96, 1, padding = 'valid', activity_regularizer = streg, use_bias = True)(eeg_input)
    block1       = Activation('relu')(block1)
    block1       = MaxPooling1D(3, strides = 2)(block1)
    
    block2       = Conv1D(128, 6, padding = 'valid', use_bias = True)(block1)
    block2       = Activation('relu')(block2)
    block2       = MaxPooling1D(3, strides = 2)(block2)
    
    block3       = Conv1D(128, 6, padding = 'valid', use_bias = True)(block2)
    block3       = Activation('relu')(block3)   
    
    flatten      = Flatten(name = 'flatten')(block3)
    dense1       = Dense(2048, activation = 'relu')(flatten)
    dense1       = Dropout(0.8)(dense1)
    dense2       = Dense(4096, activation = 'relu')(dense1)
    dense2       = Dropout(0.8)(dense2)
    output       = Dense(2)(dense2)
    softmax      = Activation('softmax', name = 'softmax')(output)
        
    return Model(inputs = eeg_input, outputs = softmax, name = 'CNNR')
