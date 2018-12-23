#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2018
#
"""
Script to save the P300 Speller database as NumPy files. Database available at
http://akimpech.izt.uam.mx/dokuwiki/doku.php?id=signal:bci:p300.es.
"""
import argparse
import sys
import numpy as np
from scipy.io import loadmat
from os import listdir
from os.path import isdir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Lista de los sujetos
subjects = ['ACS', 'APM', 'ASG', 'ASR', 'CLL', 'DCM', 'DLP', 'DMA', 'ELC', 'FSZ', 'GCE',
            'ICE', 'JCR', 'JLD', 'JLP', 'JMR', 'JSC', 'JST', 'LAC', 'LAG', 'LGP', 'LPS']

def matdir2np2(input_dir, n_channels = 6):
    """
    Function that reads directory structure of P300 Speller database from UAM
    """
    data = np.zeros((22, 2880, 206, n_channels))
    labels = np.zeros((22, 2880))
    for i,s in enumerate(subjects):
        print(input_dir + '/data' + s + '.mat')
        subject_data = loadmat(input_dir + '/data' + s + '.mat')
        data[i, :, :, :] = np.concatenate((subject_data['dataCalor'], subject_data['dataCarino'], subject_data['dataSushi']))[:, :, [0, 2, 5, 7, 8, 9]]
        if s == 'ACS':
            subject_labels = np.concatenate((subject_data['labelCalor'][:, 1:2],
                                             subject_data['labelCarino'][:, 1:2],
                                             subject_data['labelSushi'][:, 1:2]))
        else:
            subject_labels = np.concatenate((subject_data['etiqCalor'][:, 1:2],
                                             subject_data['etiqCarino'][:, 1:2],
                                             subject_data['etiqSushi'][:, 1:2]))
        np.place(subject_labels, subject_labels < 0, 0)
        labels[i, :] = np.squeeze(subject_labels)
            
    return data, labels

def matdir2np(input_dir, n_channels = 10):
    """
    Function that reads directory structure of P300 Speller database from UAM
    """
    subject_dir = listdir(input_dir)
    data = np.zeros((22, 2880, 206, n_channels))
    labels = np.zeros((22, 2880))
    i = 0
    for d in subject_dir:
        if not d.startswith('.'):
            print('Reading directory {0}'.format(d))
            subject = loadmat(input_dir + '/' + d + '/ERPgral.mat')['ERPgral']
            # Selects 6 electrodes with more information, according to Alvarado-González et al.
            if n_channels == 6:
                data[i, :, :, :] = subject.reshape((2880, 206, 10))[:, :, np.array([0, 2, 5, 7, 8, 9])]
            else:
                data[i, :, :, :] = subject.reshape((2880, 206, 10))
                
            labels[i, :] = loadmat(input_dir + '/' + d + '/etiqGral.mat')['etiqGral'][:,0]
            labels[labels == -1] = 0
            i += 1
            
    return data, labels

def main():
    """
    Main function
    """
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
            description="Saves P300 Speller database from UAM as NumPy files")
        parser.add_argument("input_dir", type=str,
                            help="Root directory of the original P300 database")
        parser.add_argument("data_filename", type=str,
                            help="Data filename")
        parser.add_argument("labels_filename", type=str,
                            help="Labels filename")
        parser.add_argument("--n_channels", type=int, default=10,
                            help="Number of electrodes to use")
        args = parser.parse_args()

        data, labels = matdir2np2(args.input_dir, n_channels = args.n_channels)
        np.save(args.data_filename, data)
        np.save(args.labels_filename, labels)
        
    except SystemExit:
        print('for help use --help')
        sys.exit(2)
        
if __name__ == "__main__":
    main()
