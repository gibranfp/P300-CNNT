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

def CNNR(Chans = 6, Samples = 206)
    
    input1   = Input(shape = (Chans, Samples))

    ##################################################################
    block1       = Conv2D(96,(1,Samples), padding = 'same')(input1)
    block1       = Activation('relu')(block1)
    block1       = MaxPooling2D((3,1), strides = 2)(block1)
    
    block1       = Conv2D(128,?, padding = 'same', use_bias = False)(block1)
    block1       = Activation('relu')(block1)
    block1       = MaxPooling2D(3, strides = 2)(block1)
    
    block1       = Conv2D(128, ?, padding = 'same', use_bias = False)(block1)
    block1       = Activation('relu')(block1)    
    
    flatten      = Flatten(name = 'flatten')(block1)
    dense        = Dense(2048,input_dim=?, activation = 'relu')(flatten)
    block1       = Dropout(0.5)(dense)
    dense        = Dense(4096,input_dim=?, activation = 'relu')(block1)
    block1       = Dropout(0.5)(dense)
    prediction   = Dense(2,activation = 'softmax')(block1)
    
    return Model(inputs=input1, outputs=prediction, name='CNNR')