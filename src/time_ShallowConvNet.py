#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2018
#
"""
Script to evaluate the CNN1 architecture (Lawhern et al., 2018) for single-trial subject-dependent P300 detection 
"""
import argparse
import sys
import numpy as np
from EEGModels import ShallowConvNet
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow import set_random_seed
from sklearn.model_selection import *
from utils import *
import tensorflow.keras.backend as K
import pandas as pd 

def evaluate_subject_model(X_train, y_train, X_valid, y_valid, X_test, y_test, timepath):
    print('X_train = {0}, X_valid = {1}, X_test = {2}'.format(X_train.shape, X_valid.shape, X_test.shape))

    n_samples = X_train.shape[1]
    n_channels = X_train.shape[2]

    sc = EEGChannelScaler(n_channels = n_channels)
    X_train = np.swapaxes(sc.fit_transform(X_train)[:, np.newaxis, :], 2, 3)
    X_valid = np.swapaxes(sc.transform(X_valid)[:, np.newaxis, :], 2, 3)
    X_test = np.swapaxes(sc.transform(X_test)[:, np.newaxis, :], 2, 3)

    model = ShallowConvNet(2, Chans = n_channels, Samples = n_samples)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

    tt = TrainTime()
    history = model.fit(X_train,
                        to_categorical(y_train),
                        batch_size = 256,
                        epochs = 10,
                        validation_data = (X_valid, to_categorical(y_valid)),
                        callbacks = [tt])

    start_test = time.time()
    proba_test = model.predict(X_test)
    test_time =  time.time() - start_test

    train_size = X_train.shape[0]
    valid_size = X_valid.shape[0]
    test_size =  X_test.shape[0]

    times = [[np.mean(tt.times), np.sum(tt.times), 10, train_size, valid_size, test_time, test_size, test_time / test_size]]
    df = pd.DataFrame(times, columns = ['Mean Epoch Time', 'Total Train Time', 'Epochs', 'Train Size', 'Valid Size', 'Test Time', 'Test Size', 'Test per example'])
    df.to_csv(timepath + 'ShallowConvNet_times.csv', encoding='utf-8')

def main():
    """
    Main function
    """
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
            description="Evaluates single-trial subject-dependent P300 detection using cross-validation")
        parser.add_argument("datatrain", type=str,
                            help="Path for the data of the P300 Speller Database (NumPy file)")
        parser.add_argument("labelstrain", type=str,
                            help="Path for the labels of the P300 Speller Database (NumPy file)")
        parser.add_argument("datavalid", type=str,
                            help="Path for the data of the P300 Speller Database (NumPy file)")
        parser.add_argument("labelsvalid", type=str,
                            help="Path for the labels of the P300 Speller Database (NumPy file)")
        parser.add_argument("datatest", type=str,
                            help="Path for the data of the P300 Speller Database (NumPy file)")
        parser.add_argument("labelstest", type=str,
                            help="Path for the labels of the P300 Speller Database (NumPy file)")
        parser.add_argument("timepath", type=str,
                            help="Path of the directory where the time is to be saved")
        args = parser.parse_args()

        np.random.seed(1)
        set_random_seed(2)

        X_train, y_train = load_db(args.datatrain, args.labelstrain)
        X_valid, y_valid = load_db(args.datavalid, args.labelsvalid)
        X_test, y_test = load_db(args.datatest, args.labelstest)
        evaluate_subject_model(X_train, y_train, X_valid, y_valid, X_test, y_test, args.timepath)

    except SystemExit:
        print('for help use --help')
        sys.exit(2)

if __name__ == "__main__":
    main()
