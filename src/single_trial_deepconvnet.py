#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2018
#
"""
Script to evaluate the P300-CNNT architecture for single-trial subject-dependent P300 detection using cross-validation
"""
import argparse
import sys
import numpy as np
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.layers import concatenate
from sklearn.utils import resample, class_weight
from sklearn.model_selection import *
from EEGModels import DeepConvNet
from utils import *

def evaluate_subject_models(data, labels, modelpath):
    """
    Trains and evaluates subject-dependent models using random cross validation.
    """
    cv = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    for i in range(data.shape[0]):
        aucs = np.zeros(10)
        accuracies = np.zeros(10)
        print("Training for subject {0}: ".format(i))
        for k, (t, v) in enumerate(cv.split(data[i], labels[i])):
            X_train, y_train, X_valid, y_valid = data[i, t], labels[i, t], data[i, v], labels[i, v]
            sample_weights = class_weight.compute_sample_weight('balanced', y_train)
                        
            pos_valid_idx = np.where(y_valid == 1)[0]
            neg_valid_idx = np.where(y_valid == 0)[0]
            usample_neg_valid_idx = np.random.choice(neg_valid_idx, len(pos_valid_idx), replace = False)
            usample_idx = np.concatenate([pos_valid_idx, usample_neg_valid_idx])
            X_valid = X_valid[usample_idx]
            y_valid = y_valid[usample_idx]

            # one hot encoding
            y_onehot_train = np.zeros((y_train.size, 2))
            y_onehot_valid = np.zeros((y_valid.size, 2))
            y_onehot_train[np.arange(y_train.size), y_train.astype(int)] = 1
            y_onehot_valid[np.arange(y_valid.size), y_valid.astype(int)] = 1
            
            sc = EEGChannelScaler()
            X_train = np.swapaxes(sc.fit_transform(X_train)[:, np.newaxis, :], 2,3)
            X_valid = np.swapaxes(sc.transform(X_valid)[:, np.newaxis, :], 2, 3)

            print('Partition {0}: X_train = {1}, X_valid = {2}'.format(k, X_train.shape, X_valid.shape))

            model = DeepConvNet(2, Chans = 6, Samples = 206)
            model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
                        
            model.fit(X_train,
                      y_onehot_train,
                      batch_size = 32,
                      sample_weight = sample_weights,
                      epochs = 50,
                      validation_data = (X_valid, y_onehot_valid))

            model.save(modelpath + '/s' + str(i) + 'p' + str(k) + '.h5')
            proba_valid = model.predict(X_valid)
            aucs[k] = roc_auc_score(y_onehot_valid, proba_valid)
            accuracies[k] = accuracy_score(y_onehot_valid, np.round(proba_valid))
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
        args = parser.parse_args()
        
        np.random.seed(0)
        data, labels = load_db(args.datapath, args.labelspath)
        evaluate_subject_models(data, labels, args.modelpath)
        
    except SystemExit:
        print('for help use --help')
        sys.exit(2)
        
if __name__ == "__main__":
    main()
