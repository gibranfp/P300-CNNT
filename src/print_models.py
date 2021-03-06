import argparse
import sys
import os.path
import tempfile
import json

from CNN1 import UCNN1, CNN1, CNN3, UCNN3
from EEGModels import ShallowConvNet, DeepConvNet, EEGNet
from BN3model import BN3
from FCNNmodel import FCNN
from OCLNN import OCLNN
from CNNR import CNNR
from SepConv1D import SepConv1D

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
        'sepconv1d_1f': (SepConv1D, {'Chans': 6, 'Samples': 206, 'Filters': 1})
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
        'sepconv1d_1f': (SepConv1D, {'Chans': 8, 'Samples': 206, 'Filters': 1})
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
        'sepconv1d_1f': (SepConv1D, {'Chans': 64, 'Samples': 156, 'Filters': 1})
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
        'sepconv1d_1f': (SepConv1D, {'Chans': 64, 'Samples': 240, 'Filters': 1})
    }
}

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
        args = parser.parse_args()

        try:
            name, params = opts[args.ds.lower()][args.arq.lower()]
            model = name(**params)
            print('Model = {0}'.format(model.get_config()))
            
        except KeyError:
            print('Unknown architecture {0}! Choose one of the following: {1}'.format(args.name, names.keys()))

    except SystemExit:
        print('for help use --help')
        sys.exit(2)

if __name__ == "__main__":
    main()
