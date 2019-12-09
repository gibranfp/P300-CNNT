#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2019
#
"""
Script to evaluate the CNNR architecture for single-trial cross-subject P300 detection
"""
import argparse
import sys
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow import set_random_seed
from sklearn.model_selection import *
from CNNR import CNNR
from utils import *
import tensorflow.keras.backend as K

def evaluate_cross_subject_model(data, labels, modelpath):
    """
    Trains and evaluates CNNR for each subject in the P300 Speller database
    using random cross validation.
    """
    n_sub = data.shape[0]
    n_ex_sub = data.shape[1]
    n_samples = data.shape[2]
    n_channels = data.shape[3]

    aucs = np.zeros(n_sub)

    data = data.reshape((n_sub * n_ex_sub, n_samples, n_channels))
    labels = labels.reshape((n_sub * n_ex_sub))
    groups = np.array([i for i in range(n_sub) for j in range(n_ex_sub)])

    cv = LeaveOneGroupOut()
    for k, (t, v) in enumerate(cv.split(data, labels, groups)):
        X_train, y_train, X_test, y_test = data[t], labels[t], data[v], labels[v]

        rg = np.random.choice(t, 1)
        sv = groups[t] == groups[rg]
        st = np.logical_not(sv)
        X_train, y_train, X_valid, y_valid = data[t][st], labels[t][st], data[t][sv], labels[t][sv]
        print("Partition {0}: train = {1}, valid = {2}, test = {3}".format(k, X_train.shape, X_valid.shape, X_test.shape))
        print("Groups train = {0}, valid = {1}, test = {2}".format(np.unique(groups[t][st]),
                                                                   np.unique(groups[t][sv]),
                                                                   np.unique(groups[v])))
            
        # channel-wise feature standarization
        sc = EEGChannelScaler(n_channels = n_channels)
        X_train = sc.fit_transform(X_train)
        X_valid = sc.transform(X_valid)
        X_test = sc.transform(X_test)
        
        model = CNNR(Chans = n_channels, Samples = n_samples)
        print(model.summary())
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

        es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 50, restore_best_weights = True)
        model.fit(X_train,
                  to_categorical(y_train),
                  batch_size = 256,
                  epochs = 200,
                  validation_data = (X_valid, to_categorical(y_valid)),
                  callbacks = [es])

        proba_test = model.predict(X_test)
        aucs[k] = roc_auc_score(y_test, proba_test[:, 1])
        print('P{0} -- AUC: {1}'.format(k, aucs[k]))
        K.clear_session()
        
    np.savetxt(modelpath + '/aucs.npy', aucs)

def main():
    """
    Main function
    """
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
            description="Evaluates single-trial cross-subject P300 detection using cross-validation")
        parser.add_argument("datapath", type=str,
                            help="Path for the data of the P300 Speller Database (NumPy file)")
        parser.add_argument("labelspath", type=str,
                            help="Path for the labels of the P300 Speller Database (NumPy file)")
        parser.add_argument("modelpath", type=str,
                            help="Path of the directory where the models are to be saved")
        args = parser.parse_args()

        np.random.seed(1)
        set_random_seed(2)
        
        data, labels = load_db(args.datapath, args.labelspath)
        evaluate_cross_subject_model(data, labels, args.modelpath)
        
    except SystemExit:
        print('for help use --help')
        sys.exit(2)
        
if __name__ == "__main__":
    main()
