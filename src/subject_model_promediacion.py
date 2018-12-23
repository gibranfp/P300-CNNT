#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2018
#
"""
Script to evaluate a simple Conv1D architecture for multiple-trial subject-dependent P300 detection using cross-validation
"""
import argparse
import sys
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import *
from resnet50_conv1d import *
from utils import *

def evaluate_subject_models_average(data, labels, modelpath, n_trials = 2):
    """
    Trains and evaluates subject-dependent models using random cross validation.
    """
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    for i in range(data.shape[0]):
        aucs = np.zeros(10)
        accuracies = np.zeros(10)
        print("Training for subject {0} with {1} trials".format(i, n_trials))
        for k, (t, v) in enumerate(cv.split(data[i], labels[i])):
            X_train, y_train, X_valid, y_valid = data[i, t], labels[i, t], data[i, v], labels[i, v]
            print('Partition {0}: X_train = {1}, X_valid = {2}'.format(k, X_train.shape, X_valid.shape))

            # averages multiple trials
            X_train, y_train = make_trial_average(X_train,
                                                  y_train,
                                                  n_trials = n_trials,
                                                  pos_samples = 5000,
                                                  neg_samples = 5000)
            X_valid, y_valid = make_trial_average(X_valid,
                                                  y_valid,
                                                  n_trials = n_trials,
                                                  pos_samples = 1000,
                                                  neg_samples = 1000)
            
            sc = EEGChannelScaler()
            X_train = sc.fit_transform(X_train)
            X_valid = sc.transform(X_valid)
            
            model = SimpleConv1D(n_channels = X_train.shape[2])
            model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            
            model.fit(X_train,
                      y_train,
                      batch_size = 256,
                      epochs = 50,
                      validation_data = (X_valid, y_valid))
            model.save(modelpath + '/s' + str(i) + 'p' + str(k) + '.h5')
            proba_valid = model.predict(X_valid)
            aucs[k] = roc_auc_score(y_valid, proba_valid)
            accuracies[k] = accuracy_score(y_valid, np.round(proba_valid))
            print('AUC: {0} ACC: {1}'.format(aucs[k], accuracies[k]))

        np.savetxt(modelpath + '/aucs_s' + str(i) + '.npy', aucs)
        np.savetxt(modelpath + '/accuracies_s' + str(i) + '.npy', accuracies)
            
def main():
    """
    Main function
    """
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
            description="Evaluates single-trial subject-dependent P300 detection using cross-validation")
        parser.add_argument("datapath", type=str,
                            help="Path for the data of the P300 Speller Database (NumPy file)")
        parser.add_argument("labelspath", type=str,
                            help="Path for the labels of the P300 Speller Database (NumPy file)")
        parser.add_argument("modelpath", type=str,
                            help="Path of the directory where the models are to be saved")
        parser.add_argument("--n_trials", type=int, default=2,
                            help="Number of trials to average")

        args = parser.parse_args()
        
        data, labels = load_db(args.datapath, args.labelspath)
        evaluate_subject_models_average(data, labels, args.modelpath, args.n_trials)
        
    except SystemExit:
        print('for help use --help')
        sys.exit(2)
        
if __name__ == "__main__":
    main()
