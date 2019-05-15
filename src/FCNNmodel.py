#!/usr/bin/env python3
# -*- coding: utf-8
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def FCNN(Chans = 6, Samples = 206):
    
    input1       = Input(shape = (Samples, Chans))
    block1       = BatchNormalization(axis = 1)(input1)
    ##################################################################
    flatten      = Flatten(name = 'flatten')(block1)
    dense        = Dense(2, activation = 'tanh',
		   	 kernel_initializer='glorot_uniform')(flatten)
    prediction   = Dense(1, activation = 'sigmoid')(dense)
    
    return Model(inputs=input1, outputs=prediction, name='FCNN')
