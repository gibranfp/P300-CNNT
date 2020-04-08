import argparse
import sys
import numpy as np
import pandas as pd
import os.path
import tempfile

from tensorflow.keras.layers import *
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications import MobileNet
from tensorflow.compat.v1 import graph_util

from CNN1 import UCNN1, CNN1, CNN3, UCNN3
from EEGModels import ShallowConvNet, DeepConvNet, EEGNet
from BN3model import BN3
from FCNNmodel import FCNN
from OCLNN import OCLNN
from CNNR import CNNR
from SepConv1D import SepConv1D
from SepConv1D_Ext import SepConv1DExt

opts = {
    'uam': {
        'bn3': (BN3, {'Chans': 6, 'Samples': 206}),
        'shallowconvnet': (ShallowConvNet, {'nb_classes': 2, 'Chans': 6, 'Samples': 206}),
        'deepconvnet': (DeepConvNet, {'nb_classes': 2, 'Chans': 6, 'Samples': 206}),
        'eegnet': (EEGNet, {'nb_classes': 2, 'Chans': 6, 'Samples': 206}),
        'cnn1': (CNN1, {'Chans': 6, 'Samples': 206}),
        'ucnn1': (UCNN1, {'Chans': 6, 'Samples': 206}),
        'cnn3': (CNN3, {'Chans': 6, 'Samples': 206}),
        'ucnn3': (UCNN3, {'Chans': 6, 'Samples': 206}),
        'cnnr': (CNNR, {'Chans': 6, 'Samples': 206}),
        'fcnn': (FCNN, {'input_dim': 6 * 206}),
        'oclnn': (OCLNN, {'Chans': 6, 'Samples': 206, 'SegmentSize': 14, 'Padding': 2}),
        'sepconv1d_32f': (SepConv1D, {'Chans': 6, 'Samples': 206, 'Filters': 32}),
        'sepconv1d_16f': (SepConv1D, {'Chans': 6, 'Samples': 206, 'Filters': 16}),
        'sepconv1d_8f': (SepConv1D, {'Chans': 6, 'Samples': 206, 'Filters': 8}),
        'sepconv1d_4f': (SepConv1D, {'Chans': 6, 'Samples': 206, 'Filters': 4}),
        'sepconv1d_2f': (SepConv1D, {'Chans': 6, 'Samples': 206, 'Filters': 2}),
        'sepconv1d_1f': (SepConv1D, {'Chans': 6, 'Samples': 206, 'Filters': 1}),
        'sepconv1dext_32f': (SepConv1DExt, {'Chans': 6, 'Samples': 206, 'Filters': 32}),
        'sepconv1dext_16f': (SepConv1DExt, {'Chans': 6, 'Samples': 206, 'Filters': 16}),
        'sepconv1dext_8f': (SepConv1DExt, {'Chans': 6, 'Samples': 206, 'Filters': 8}),
        'sepconv1dext_4f': (SepConv1DExt, {'Chans': 6, 'Samples': 206, 'Filters': 4}),
        'sepconv1dext_2f': (SepConv1DExt, {'Chans': 6, 'Samples': 206, 'Filters': 2}),
        'sepconv1dext_1f': (SepConv1DExt, {'Chans': 6, 'Samples': 206, 'Filters': 1})
    },
    'horizon': {
        'bn3': (BN3, {'Chans': 8, 'Samples': 206}),
        'shallowconvnet': (ShallowConvNet, {'nb_classes': 2, 'Chans': 8, 'Samples': 206}),
        'deepconvnet': (DeepConvNet, {'nb_classes': 2, 'Chans': 8, 'Samples': 206}),
        'eegnet': (EEGNet, {'nb_classes': 2, 'Chans': 8, 'Samples': 206}),
        'cnn1': (CNN1, {'Chans': 8, 'Samples': 206}),
        'ucnn1': (UCNN1, {'Chans': 8, 'Samples': 206}),
        'cnn3': (CNN3, {'Chans': 8, 'Samples': 206}),
        'ucnn3': (UCNN3, {'Chans': 8, 'Samples': 206}),
        'cnnr': (CNNR, {'Chans': 8, 'Samples': 206}),
        'fcnn': (FCNN, {'input_dim': 8 * 206}),
        'oclnn': (OCLNN, {'Chans': 8, 'Samples': 206, 'SegmentSize': 14, 'Padding': 2}),
        'sepconv1d_32f': (SepConv1D, {'Chans': 8, 'Samples': 206, 'Filters': 32}),
        'sepconv1d_16f': (SepConv1D, {'Chans': 8, 'Samples': 206, 'Filters': 16}),
        'sepconv1d_8f': (SepConv1D, {'Chans': 8, 'Samples': 206, 'Filters': 8}),
        'sepconv1d_4f': (SepConv1D, {'Chans': 8, 'Samples': 206, 'Filters': 4}),
        'sepconv1d_2f': (SepConv1D, {'Chans': 8, 'Samples': 206, 'Filters': 2}),
        'sepconv1d_1f': (SepConv1D, {'Chans': 8, 'Samples': 206, 'Filters': 1}),
        'sepconv1dext_32f': (SepConv1DExt, {'Chans': 8, 'Samples': 206, 'Filters': 32}),
        'sepconv1dext_16f': (SepConv1DExt, {'Chans': 8, 'Samples': 206, 'Filters': 16}),
        'sepconv1dext_8f': (SepConv1DExt, {'Chans': 8, 'Samples': 206, 'Filters': 8}),
        'sepconv1dext_4f': (SepConv1DExt, {'Chans': 8, 'Samples': 206, 'Filters': 4}),
        'sepconv1dext_2f': (SepConv1DExt, {'Chans': 8, 'Samples': 206, 'Filters': 2}),
        'sepconv1dext_1f': (SepConv1DExt, {'Chans': 8, 'Samples': 206, 'Filters': 1})
    },
    'bci_ii': {
        'bn3': (BN3, {'Chans': 64, 'Samples': 156}),
        'shallowconvnet': (ShallowConvNet, {'nb_classes': 2, 'Chans': 64, 'Samples': 156}),
        'deepconvnet': (DeepConvNet, {'nb_classes': 2, 'Chans': 64, 'Samples': 156}),
        'eegnet': (EEGNet, {'nb_classes': 2, 'Chans': 64, 'Samples': 156}),
        'cnn1': (CNN1, {'Chans': 64, 'Samples': 156}),
        'ucnn1': (UCNN1, {'Chans': 64, 'Samples': 156}),
        'cnn3': (CNN3, {'Chans': 64, 'Samples': 156}),
        'ucnn3': (UCNN3, {'Chans': 64, 'Samples': 156}),
        'cnnr': (CNNR, {'Chans': 64, 'Samples': 156}),
        'fcnn': (FCNN, {'input_dim': 64 * 156}),
        'oclnn': (OCLNN, {'Chans': 64, 'Samples': 156, 'SegmentSize': 11, 'Padding': (5, 4)}),
        'sepconv1d_32f': (SepConv1D, {'Chans': 64, 'Samples': 156, 'Filters': 32}),
        'sepconv1d_16f': (SepConv1D, {'Chans': 64, 'Samples': 156, 'Filters': 16}),
        'sepconv1d_8f': (SepConv1D, {'Chans': 64, 'Samples': 156, 'Filters': 8}),
        'sepconv1d_4f': (SepConv1D, {'Chans': 64, 'Samples': 156, 'Filters': 4}),
        'sepconv1d_2f': (SepConv1D, {'Chans': 64, 'Samples': 156, 'Filters': 2}),
        'sepconv1d_1f': (SepConv1D, {'Chans': 64, 'Samples': 156, 'Filters': 1}),
        'sepconv1dext_32f': (SepConv1DExt, {'Chans': 64, 'Samples': 156, 'Filters': 32}),
        'sepconv1dext_16f': (SepConv1DExt, {'Chans': 64, 'Samples': 156, 'Filters': 16}),
        'sepconv1dext_8f': (SepConv1DExt, {'Chans': 64, 'Samples': 156, 'Filters': 8}),
        'sepconv1dext_4f': (SepConv1DExt, {'Chans': 64, 'Samples': 156, 'Filters': 4}),
        'sepconv1dext_2f': (SepConv1DExt, {'Chans': 64, 'Samples': 156, 'Filters': 2}),
        'sepconv1dext_1f': (SepConv1DExt, {'Chans': 64, 'Samples': 156, 'Filters': 1})
    },
    'bci_iii': {
        'bn3': (BN3, {'Chans': 64, 'Samples': 240}),
        'shallowconvnet': (ShallowConvNet, {'nb_classes': 2, 'Chans': 64, 'Samples': 240}),
        'deepconvnet': (DeepConvNet, {'nb_classes': 2, 'Chans': 64, 'Samples': 240}),
        'eegnet': (EEGNet, {'nb_classes': 2, 'Chans': 64, 'Samples': 240}),
        'cnn1': (CNN1, {'Chans': 64, 'Samples': 240}),
        'ucnn1': (UCNN1, {'Chans': 64, 'Samples': 240}),
        'cnn3': (CNN3, {'Chans': 64, 'Samples': 240}),
        'ucnn3': (UCNN3, {'Chans': 64, 'Samples': 240}),
        'cnnr': (CNNR, {'Chans': 64, 'Samples': 240}),
        'fcnn': (FCNN, {'input_dim': 64 * 240}),
        'oclnn': (OCLNN, {'Chans': 64, 'Samples':240, 'SegmentSize': 16, 'Padding': 0}),
        'sepconv1d_32f': (SepConv1D, {'Chans': 64, 'Samples': 240, 'Filters': 32}),
        'sepconv1d_16f': (SepConv1D, {'Chans': 64, 'Samples': 240, 'Filters': 16}),
        'sepconv1d_8f': (SepConv1D, {'Chans': 64, 'Samples': 240, 'Filters': 8}),
        'sepconv1d_4f': (SepConv1D, {'Chans': 64, 'Samples': 240, 'Filters': 4}),
        'sepconv1d_2f': (SepConv1D, {'Chans': 64, 'Samples': 240, 'Filters': 2}),
        'sepconv1d_1f': (SepConv1D, {'Chans': 64, 'Samples': 240, 'Filters': 1}),
        'sepconv1dext_32f': (SepConv1DExt, {'Chans': 64, 'Samples': 240, 'Filters': 32}),
        'sepconv1dext_16f': (SepConv1DExt, {'Chans': 64, 'Samples': 240, 'Filters': 16}),
        'sepconv1dext_8f': (SepConv1DExt, {'Chans': 64, 'Samples': 240, 'Filters': 8}),
        'sepconv1dext_4f': (SepConv1DExt, {'Chans': 64, 'Samples': 240, 'Filters': 4}),
        'sepconv1dext_2f': (SepConv1DExt, {'Chans': 64, 'Samples': 240, 'Filters': 2}),
        'sepconv1dext_1f': (SepConv1DExt, {'Chans': 64, 'Samples': 240, 'Filters': 1})
    }
}

def trainable(model):
    return int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))

def nontrainable(model):
    return int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=K.get_session().graph,
                                          run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops

def main():
    """
    Main function
    """
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
            description="Prints statistics for different architectures")
        parser.add_argument("ds", type=str,
                            help="Name of the dataset")
        parser.add_argument("arq", type=str,
                            help="Name of the architecture")
        parser.add_argument("dirpath", type=str,
                            help="Name of the architecture")
        args = parser.parse_args()

        try:
            name, params = opts[args.ds.lower()][args.arq.lower()]
            model = name(**params)
            with open(args.dirpath + '/' + args.arq + '_summary.txt','w') as f:
                model.summary(print_fn = lambda s: f.write(s + '\n'))

            complexity = [[model.count_params(), trainable(model), nontrainable(model), get_flops(model)]]
            df = pd.DataFrame(complexity, columns = ['#Params', '#Trainable', '#NonTrainable', 'FLOPS'])
            df.to_csv(args.dirpath + '/' + args.arq + '_statistics.csv', encoding='utf-8')

        except KeyError:
            print('Unknown architecture {0}! Choose one of the following: {1}'.format(args.name, names.keys()))

    except SystemExit:
        print('for help use --help')
        sys.exit(2)

if __name__ == "__main__":
    main()
