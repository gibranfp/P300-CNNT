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
from tensorflow.keras import backend as K

def create_base_network():
  eeg_input = Input(shape=(206,1))

  x = Conv1D(64, 16, padding = 'same')(eeg_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling1D(2, strides = 2)(x)
  x = Dropout(0.5)(x)
  x = Conv1D(128, 16, padding = 'same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling1D(2, strides = 2)(x)
  x = Dropout(0.5)(x)
  x = Flatten()(x)

  return Model(eeg_input, x)

def P300_CNNT(activation = 'relu', pad = 'same', n_channels = 6):
  base_net = create_base_network()
  data_input = Input(shape=(206, n_channels))
  branch_outputs = []
  for c in range(n_channels):
    branch_in = Lambda(lambda x: K.expand_dims(x[:, :, c], -1))(data_input)
    out = base_net(branch_in)
    branch_outputs.append(out)
    
  all_ch = Add()(branch_outputs)
  x = Dense(128, activation = 'relu')(all_ch)
  x = Dropout(0.5)(x)
  prediction = Dense(1, activation = 'sigmoid')(all_ch)

  return Model(data_input, prediction, name='p300-cnnt')  
