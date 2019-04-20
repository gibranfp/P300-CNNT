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


def FCNN(Chans = 6, Samples = 206, Ns=16):
    
    input1       = Input(shape = (Samples, Chans))
    block1       = BatchNormalization(axis = 1)(input1)
    ##################################################################
    dense        = Dense(2, activation = 'tanh',
		   	 kernel_initializer='glorot_uniform')(block1)
    prediction   = Dense(1, activation = 'sigmoid')(block1)
    
    return Model(inputs=input1, outputs=prediction, name='FCNN')
