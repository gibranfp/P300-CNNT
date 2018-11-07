#!/usr/bin/env python3
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2018
#
"""
Script to compute mean AUC from directory
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

subjects = ['ACS', 'APM', 'ASG', 'ASR', 'CLL', 'DCM', 'DLP', 'DMA', 'ELC', 'FSZ', 'GCE',
            'ICE', 'JCR', 'JLD', 'JLP', 'JMR', 'JSC', 'JST', 'LAC', 'LAG', 'LGP', 'LPS']

def plot_aucs(aucpath, filepath):
    """
    Plot AUCs
    """
    data_to_plot = []
    aucs = np.zeros((22,10))
    for i in range(22):
        aucs[i, :] = np.loadtxt(aucpath + '/aucs_s' + str(i) + '.npy')
        data_to_plot.append(np.loadtxt(aucpath + '/aucs_s' + str(i) + '.npy'))

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot, showmeans=True, meanline=True)
    ax.set_xticklabels(subjects, rotation=45)
    plt.ylabel("AUC")
    plt.xlabel('Subjects')
    plt.grid(False)
    plt.savefig(filepath)
    
def main():
    """
    Main function
    """
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
            description="Plot AUCs")
        parser.add_argument("aucpath", type=str,
                            help="Path of the directory where the AUCs are saved")
        parser.add_argument("filepath", type=str,
                            help="File path where the plot are to be saved")
        args = parser.parse_args()
        
        plot_aucs(args.aucpath, args.filepath)
        
    except SystemExit:
        print('for help use --help')
        sys.exit(2)
        
if __name__ == "__main__":
    main()
