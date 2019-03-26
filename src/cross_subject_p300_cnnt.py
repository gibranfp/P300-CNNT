#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2018
#
"""
Script to evaluate a ResNet50 architecture for single-trial cross_subject P300 detection using cross-validation
"""
import argparse
import sys
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import *
from sklearn.utils import resample, class_weight
from p300_cnnt  import P300_CNNT
from utils import *

def evaluate_cross_subject_model(data, labels, modelpath):
    """
    Trains and evaluates SimpleConv1D architecture for each subject in the P300 Speller database
    using random cross validation.
    """
    aucs = np.zeros(22)
    accuracies = np.zeros(22)
    data = data.reshape((22 * 2880, 206, data.shape[3]))
    labels = labels.reshape((22 * 2880))
    groups = [i for i in range(22) for j in range(2880)]
    cv = LeaveOneGroupOut()
    for k, (t, v) in enumerate(cv.split(data, labels, groups)):
        X_train, y_train, X_test, y_test = data[t], labels[t], data[v], labels[v]

        print("Partition {0}: train = {1}, valid = {2}".format(k, X_train.shape, X_test.shape))
        sample_weights = class_weight.compute_sample_weight('balanced', y_train)

        sc = EEGChannelScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        model = P300_CNNT(n_channels = X_train.shape[2])
        print(model.summary())
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=50)
        model.fit(X_train,
                  y_train,
                  batch_size = 32,
                  sample_weight = sample_weights,
                  epochs = 200,
                  validation_split = 0.2,
                  callbacks = [es])

        model.save(modelpath + '/s' + str(k) + '.h5')
        proba_test = model.predict(X_test)
        aucs[k] = roc_auc_score(y_test, proba_test)
        accuracies[k] = accuracy_score(y_test, np.round(proba_test))
        print('AUC: {0} ACC: {1}'.format(aucs[k], accuracies[k]))
    np.savetxt(modelpath + '/aucs.npy', aucs)
    np.savetxt(modelpath + '/accuracies.npy', accuracies)

def main():
    """
    Main function
    """
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
            description="Evaluates single-trial cross_subject P300 detection using cross-validation")
        parser.add_argument("datapath", type=str,
                            help="Path for the data of the P300 Speller Database (NumPy file)")
        parser.add_argument("labelspath", type=str,
                            help="Path for the labels of the P300 Speller Database (NumPy file)")
        parser.add_argument("modelpath", type=str,
                            help="Path of the directory where the models are to be saved")
        args = parser.parse_args()
        
        data, labels = load_db(args.datapath, args.labelspath)
        evaluate_cross_subject_model(data, labels, args.modelpath)
        
    except SystemExit:
        print('for help use --help')
        sys.exit(2)
        
if __name__ == "__main__":
    main()
