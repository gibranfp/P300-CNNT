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
import os

def compute_mean_auc(aucpath, title):
    """
    Compute mean AUC
    """
    aucs = np.zeros((22,50))
    for i in range(22):
        aucs[i, :] = np.loadtxt(aucpath + '/s' + str(i) + '_aucs.npy')
        
    print('---------' + title + '---------')
    print('Average Mean AUC: {0}'.format(aucs.mean()))
    print('std(mean(aucs_per_subject)): {0}'.format(aucs.mean(axis = 1).std()))
    print('std(all_aucs): {0}'.format(aucs.std()))
    print('mean(std(aucs_per_subject)): {0}'.format(aucs.std(axis = 1).mean()))

def main():
    """
    Main function
    """
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
            description="Compute mean AUC")
        parser.add_argument("aucpath", type=str,
                            help="Path of the directory where the AUCs are saved")
        parser.add_argument("title", type=str,
                            help="Title")
        args = parser.parse_args()
        
        compute_mean_auc(args.aucpath, args.title)
        
    except SystemExit:
        print('for help use --help')
        sys.exit(2)
        
if __name__ == "__main__":
    main()
