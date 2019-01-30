#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2018
#
"""
Functions to read P300 Speller database as NumPy files. 
"""
import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class roc_callback(tf.keras.callbacks.Callback):
    """
    Class for Keras callback to calculate ROC curve and AUC
    """
    
    def __init__(self,training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        
    def on_train_begin(self, logs={}):
        return
        
    def on_train_end(self, logs={}):
        return
        
    def on_epoch_begin(self, epoch, logs={}):
        return
        
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        
        print('AUC Train: {0} AUC Valid: {1}'.format(roc, roc_val))
        
        return
        
    def on_batch_begin(self, batch, logs={}):
        return
        
    def on_batch_end(self, batch, logs={}):
        return
    
def load_db(datafile, labelsfile):
    """
    Function that reads P300 Speller database from UAM
    """
    print('Reading P300 Speller database')
    data = np.load(datafile)
    labels = np.load(labelsfile)
    print('Data shape = {0}, Labels shape = {1}'.format(data.shape, labels.shape))
    
    return data, labels

class EEGChannelScaler():
    """
    Class to scale each channel of an EEG signal separately
    """
    def __init__(self, n_channels = 6):
        """
        Defines and initialize a standard scaler for each channel
        """
        self.n_channels_ = n_channels
        self.sc_ = []
        for c in range(self.n_channels_):
            self.sc_.append(StandardScaler())
            
    def fit_transform(self, X):
        """
        Fits the standard scaler of scikit-learn for each channel using the training data
        """
        if (X.shape[2] != self.n_channels_):
            print('Error: Expected {0} channels, got {1} instead.'.format(self.n_channels_, X.shape[2]))

        for c in range(self.n_channels_):
            X[:, :, c] = self.sc_[c].fit_transform(X[:, :, c])

        self.fitted_ = True
        
        return X
    
    def transform(self, X):
        """
        Scales an array (nrows, ncols, nchannels) for each channel separately
        """
        for c in range(self.n_channels_):
            X[:, :, c] = self.sc_[c].transform(X[:, :, c])

        return X

def make_trial_average(X, y, n_trials = 2, pos_samples = 10000, neg_samples = 10000):
    """
    Function that makes trial averages
    """
    print('Generating {0} positive and {1} negative samples of {2}-trial averages'.format(pos_samples, neg_samples, n_trials))

    X_pos = X[y == 1, :, :]
    X_neg = X[y == 0, :, :]
    
    X_avg = np.zeros((pos_samples + neg_samples, X.shape[1], X.shape[2]))
    y_avg = np.zeros(pos_samples + neg_samples)
    y_avg[:pos_samples] = 1
    for i in range(pos_samples):
        pos_trials = np.random.choice(X_pos.shape[0], n_trials)
        X_avg[i, :, :] = np.mean(X_pos[pos_trials, :, :], axis = 0)

    for i in range(neg_samples):
        neg_trials = np.random.choice(X_neg.shape[0], n_trials)
        X_avg[pos_samples + i, :, :] = np.mean(X_neg[neg_trials, :, :], axis = 0)
            
    perm = np.random.permutation(pos_samples + neg_samples)
    X_avg = X_avg[perm, :, :]
    y_avg = y_avg[perm]
        
    return X_avg, y_avg
        
def stack_trials(X, y, n_trials = 2, pos_samples = 1000, neg_samples = 1000):
    """
    Function that makes trial averages
    """
    print('X shape = {0}, y shape = {1}'.format(X.shape, y.shape))
    print('Generating {0} positive and {1} negative samples of {2}-trial averages'.format(pos_samples, neg_samples, n_trials))
    
    X_pos = X[y == 1, :, :]
    X_neg = X[y == 0, :, :]
    
    X_stack = np.zeros((pos_samples + neg_samples, X.shape[1], X.shape[2], n_trials))
    y_stack = np.zeros(pos_samples + neg_samples)
    for i in range(pos_samples):
        pos_trials = np.random.choice(X_pos.shape[0], n_trials)
        X_stack[i, :, :, :] = X_pos[pos_trials, :, :].transpose(1, 2, 0)
        y_stack[i] = 1
        
        for i in range(neg_samples):
            neg_trials = np.random.choice(X_neg.shape[0], n_trials)
            X_stack[pos_samples + i, :, :, :] = X_neg[neg_trials, :, :].transpose(1, 2, 0)
            y_stack[pos_samples + i] = 0
            
            perm = np.random.permutation(pos_samples + neg_samples)
            X_stack = X_stack[perm, :, :]
            y_stack = y_stack[perm]
            
    return X_stack, y_stack
        
def balance_data(X, y, n_samples = 1000, btype = 'downsample'):
    """
    Function to balance data
    """
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    if btype == 'downsample':
        X_neg = resample(X_neg, replace = False, n_samples = n_samples)
    else:
        X_pos = resample(X_pos, replace = True, n_samples = n_samples)

    X_balanced = np.concatenate((X_neg, X_pos))
    y_balanced = np.zeros(X_neg.shape[0] + X_pos.shape[0])
    y_balanced[X_neg.shape[0]:] = 1
    perm = np.random.permutation(X_balanced.shape[0])
    
    return X_balanced[perm], y_balanced[perm]
