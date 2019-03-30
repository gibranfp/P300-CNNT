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

def CNN1(Chans = 6,Samples=206):
    input1       = Input(shape = (Chans, Samples))
    ##################################################################
    block1       = Conv2D(?, ?, padding = 'same',
                                   input_shape = (Chans, Samples),
                                   use_bias = False)(input1)
    block1       = Activation('tanh')(block1)
    block1       = Conv2D(?, (1, 128), padding = 'same',
                                   input_shape = (Chans, Samples),
                                   use_bias = False)(input1)
    block1       = Activation('tanh')(block1)
    flatten      = Flatten(name = 'flatten')(block1)
    dense        = Dense(100,input_dim=Chans, activation = 'sigmoid')(flatten)
    prediction   = Dense(2, activation = 'sigmoid')(dense)
  return Model(inputs=input1, outputs=prediction, name='CNN1')
