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
from keras.utils.generic_utils import get_custom_objects

def scaled_tanh(z):
    '''
    Scaled hyperbolic tangent activation function, as proposed
    by Lecun 1989:
    http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf

    See also Lecun et al. 1998:
    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf 
    '''
    return 1.7159 * K.tanh((2.0 / 3.0) * z)

get_custom_objects().update({'scaled_tanh': Activation(scaled_tanh)})

def CNN1(Chans = 6, Samples = 206):
    eeg_input    = Input(shape = (Samples, Chans))
    
    block1       = Conv1D(10, 1, padding = 'same',
                          data_format = 'channels_last',
                          bias_initializer = 'glorot_uniform',
                          kernel_initializer = 'glorot_uniform',
                          use_bias = True)(eeg_input)
    block1       = Activation('scaled_tanh')(block1)

    block1       = Conv1D(50, 13, padding = 'same',
                          data_format = 'channels_last',
                          bias_initializer = 'glorot_uniform',
                          kernel_initializer = 'glorot_uniform',
                          use_bias = True)(block1)
    block1       = Activation('scaled_tanh')(block1)
    
    flatten      = Flatten(name = 'flatten')(block1)
    dense        = Dense(100, activation = 'sigmoid')(flatten)
    dense        = Dense(2)(dense)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs = eeg_input, outputs = softmax, name = 'CNN1')
