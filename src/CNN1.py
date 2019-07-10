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
from tensorflow.keras.utils import get_custom_objects

def cecotti_normal(shape, dtype = None, partition_info = None):
    '''
    Initializer proposed by Cecotti et al. 2011:
    https://ieeexplore.ieee.org/document/5492691
    '''
    if len(shape) == 1:
        fan_in = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
    else:
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
  
    return K.random_normal(shape, mean = 0.0, stddev = (1.0 / fan_in))

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
                          bias_initializer = cecotti_normal,
                          kernel_initializer = cecotti_normal,
                          use_bias = True)(eeg_input)
    block1       = Activation('scaled_tanh')(block1)

    block1       = Conv1D(50, 13, padding = 'same',
                          data_format = 'channels_last',
                          bias_initializer = cecotti_normal,
                          kernel_initializer = cecotti_normal,
                          use_bias = True)(block1)
    block1       = Activation('scaled_tanh')(block1)
    
    flatten      = Flatten(name = 'flatten')(block1)
    dense        = Dense(100, activation = 'sigmoid')(flatten)
    prediction   = Dense(2, activation = 'sigmoid')(dense)
    
    return Model(inputs = eeg_input, outputs = prediction, name = 'CNN1')

def UCNN1(Chans = 6, Samples = 206):
    eeg_input    = Input(shape = (Samples, Chans))
    
    block1       = Conv1D(10, 1, padding = 'same',
                          data_format = 'channels_last',
                          bias_initializer = cecotti_normal,
                          kernel_initializer = cecotti_normal,
                          use_bias = True)(eeg_input)
    block1       = Activation('scaled_tanh')(block1)

    block1       = Conv1D(50, 13, padding = 'same',
                          data_format = 'channels_last',
                          bias_initializer = cecotti_normal,
                          kernel_initializer = cecotti_normal,
                          use_bias = True)(block1)
    block1       = Activation('scaled_tanh')(block1)
    
    flatten      = Flatten(name = 'flatten')(block1)
    dense        = Dense(100, activation = 'sigmoid')(flatten)
    dense        = Dense(2)(dense)
    softmax      = Activation('softmax', name = 'softmax')(dense)
        
    return Model(inputs = eeg_input, outputs = softmax, name = 'UCNN1')

def CNN3(Chans = 6, Samples = 206):
    eeg_input    = Input(shape = (Samples, Chans))
    
    block1       = Conv1D(1, 1, padding = 'same',
                          data_format = 'channels_last',
                          bias_initializer = cecotti_normal,
                          kernel_initializer = cecotti_normal,
                          use_bias = True)(eeg_input)
    block1       = Activation('scaled_tanh')(block1)

    block1       = Conv1D(50, 13, padding = 'same',
                          data_format = 'channels_last',
                          bias_initializer = cecotti_normal,
                          kernel_initializer = cecotti_normal,
                          use_bias = True)(block1)
    block1       = Activation('scaled_tanh')(block1)
    
    flatten      = Flatten(name = 'flatten')(block1)
    dense        = Dense(100, activation = 'sigmoid')(flatten)
    prediction   = Dense(2, activation = 'sigmoid')(dense)
    
    return Model(inputs = eeg_input, outputs = prediction, name = 'CNN1')

def UCNN3(Chans = 6, Samples = 206):
    eeg_input    = Input(shape = (Samples, Chans))
    
    block1       = Conv1D(1, 1, padding = 'same',
                          data_format = 'channels_last',
                          bias_initializer = cecotti_normal,
                          kernel_initializer = cecotti_normal,
                          use_bias = True)(eeg_input)
    block1       = Activation('scaled_tanh')(block1)

    block1       = Conv1D(50, 13, padding = 'same',
                          data_format = 'channels_last',
                          bias_initializer = cecotti_normal,
                          kernel_initializer = cecotti_normal,
                          use_bias = True)(block1)
    block1       = Activation('scaled_tanh')(block1)
    
    flatten      = Flatten(name = 'flatten')(block1)
    dense        = Dense(100, activation = 'sigmoid')(flatten)
    dense        = Dense(2)(dense)
    softmax      = Activation('softmax', name = 'softmax')(dense)
        
    return Model(inputs = eeg_input, outputs = softmax, name = 'UCNN1')
