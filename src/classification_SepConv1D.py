#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2019
#
"""
Script to evaluate the P300-CNNT architecture for single-trial subject-dependent P300 detection 
"""
import argparse
import sys
import numpy as np
from SepConv1D import SepConv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import set_random_seed
from sklearn.model_selection import *
from utils import *
import tensorflow.keras.backend as K

def evaluate_subject_models(data, labels, modelpath, subject, n_filters = 32):
    """
    Trains and evaluates P300-CNNT for each subject in the P300 Speller database
    using repeated stratified K-fold cross validation.
    """
    
    n_samples = data.shape[2]
    n_channels = data.shape[3]


      model = SepConv1D(Chans = n_channels, Samples = n_samples, Filters = n_filters)
      model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

      es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 50, restore_best_weights = True)
      history = model.fit(X_train,
                          y_train,
                          batch_size = 256,
                          epochs = 200,
                          validation_data = (X_valid, y_valid),
                          callbacks = [es])

      model.load_weights('model.h5')
      predictions_single = model.predict(X_test)
      prediction_result = np.argmax(predictions_single[0])
      print(prediction_result)
            
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
        parser.add_argument("--subject", type=int,
                            help="Subject to evaluate")
        parser.add_argument("--n_filters", type=int, default = 32,
                            help="Number of filters to use for SepConv1D")

        args = parser.parse_args()
        
        data, labels = load_db(args.datapath, args.labelspath)
        evaluate_subject_models(data, labels, args.modelpath, args.subject, n_filters = args.n_filters)
        
    except SystemExit:
        print('for help use --help')
        sys.exit(2)
        
if __name__ == "__main__":
    main()
