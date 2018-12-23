#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2018
#
"""
Script to evaluate a fine tuned ResNet50 architecture for single trial subject-dependent P300 detection using cross-validation 
"""
import argparse
import sys
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import *
from resnet50_conv1d import *
from utils import *

def evaluate_finetuned_subject_models(data, labels, base_modelpath, finetuned_modelpath):
    """
    Trains and evaluates fine-tuned single-trial subject-dependent models
    """
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    for i in range(data.shape[0]):
        aucs = np.zeros(10)
        print("Training for subject {0}: ".format(i))
        for k, (t, v) in enumerate(cv.split(data[i], labels[i])):
            X_train, y_train, X_valid, y_valid = data[i, t], labels[i, t], data[i, v], labels[i, v]
            print('Partition {0}: X_train = {1}, X_valid = {2}'.format(k, X_train.shape, X_valid.shape))

            # normalizes each channel separately
            for j in range(X_train.shape[2]):
                sc = StandardScaler()
                X_train[:, :, j] = sc.fit_transform(X_train[:, :, j])
                X_valid[:, :, j] = sc.transform(X_valid[:, :, j])

            base_model = ResNet50(weights = base_modelpath + '/', input_shape = (206, X_train.shape[2]))
            base_model.layers.pop()
            for layer in base_model.layers:
                layer.trainable = False
            last = base_model.layers[-1].output
            model = Dense(1, activation='sigmoid')(last)
            
            finetuned_model = Model(base_model.input, model)
            finetuned_model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
            finetuned_model.fit(X_train,
                                y_train,
                                batch_size = 256,
                                epochs = 50)
            
            finetuned_model.save(finetuned_modelpath + '/finetuned_s' + str(i) + 'p' + str(k) + '.h5')
            proba_valid = finetuned_model.predict(X_valid)
            fpr, tpr, thresholds = roc_curve(y_valid, proba_valid, pos_label=1)
            aucs[k] = auc(fpr, tpr)
            
        np.savetxt(finetuned_modelpath + '/aucs_finetuned_s' + str(i) + '.npy', aucs)

def main():
    """
    Main function
    """
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
            description="Evaluates single-trial subject-dependent P300 detection with fine tuning using cross-validation")
        parser.add_argument("datapath", type=str,
                            help="Path for the data of the P300 Speller Database (NumPy file)")
        parser.add_argument("labelspath", type=str,
                            help="Path for the labels of the P300 Speller Database (NumPy file)")
        parser.add_argument("base_dirpath", type=str,
                            help="Directory path where the independent base models are saved")
        parser.add_argument("finetued_path", type=str,
                            help="Directory path where the fine tuned subject models are to be saved")

        args = parser.parse_args()
        
        data, labels = load_db(args.datapath, args.labelspath)
        evaluate_finetuned_subject_models(data, labels, args.modelpath)
        
    except SystemExit:
        print('for help use --help')
        sys.exit(2)
        
if __name__ == "__main__":
    main()
