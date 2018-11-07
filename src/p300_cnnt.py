#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2018
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def P300_CNNT(activation = 'relu', pad = 'same', n_channels = 6):
  eeg_input = Input(shape=(206, n_channels))
  
  model = Conv1D(64, 8, padding = 'same')(eeg_input)
  model = BatchNormalization()(model)
  model = Activation('relu')(model)
  model = MaxPooling1D(2, strides = 2)(model)
  model = Dropout(0.5)(model)

  model = Conv1D(128, 8, padding = 'same')(model)
  model = BatchNormalization()(model)
  model = Activation('relu')(model)
  model = MaxPooling1D(2, strides = 2)(model)
  model = Dropout(0.5)(model)

  model = Flatten()(model)
  model = Dense(256, activation = 'relu')(model)
  model = Dropout(0.5)(model)
  model = Dense(128, activation = 'relu')(model)
  model = Dropout(0.5)(model)
  model = Dense(1, activation = 'sigmoid')(model)

  return Model(eeg_input, model, name='p300-cnnt')
