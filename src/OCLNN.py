#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2019
#
'''
Keras implementation of the One Convolution Layer Neural Network (OCLNN)
proposed by Shan et al. 2018: https://www.ijcai.org/proceedings/2018/0222.pdf
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def OCLNN(Chans = 6, Samples = 206):
    eeg_input    = Input(shape = (Samples, Chans))

    padded       = ZeroPadding1D(padding = 2)(eeg_input)
    block1       = Conv1D(16, 14, strides = 14,
                          padding = 'valid',
                          data_format = 'channels_last',
                          kernel_initializer = 'glorot_uniform',
                          bias_initializer = 'zeros',
                          kernel_regularizer = l2(l = 0.01),
                          bias_regularizer = l2(l = 0.01),
                          use_bias = True)(padded)
    block1       = Activation('relu')(block1)
    block1       = Dropout(0.25)(block1)
    flatten      = Flatten(name = 'flatten')(block1)
    dense        = Dense(2)(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs = eeg_input, outputs = softmax, name = 'OCLNN')
