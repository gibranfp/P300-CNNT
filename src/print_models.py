import argparse
import sys
from CNN1 import UCNN1, CNN1, CNN3, UCNN3
from EEGModels import ShallowConvNet, DeepConvNet, EEGNet
from BN3model import BN3
from FCNNmodel import FCNN
from OCLNN import OCLNN
from CNNR import CNNR
from SepConv1D import SepConv1D

names = {
    'bn3': (BN3, {}),
    'shallowconvnet': (ShallowConvNet, {'nb_classes': 2, 'Chans': 6, 'Samples': 206}),
    'deepconvnet': (DeepConvNet, {'nb_classes': 2, 'Chans': 6, 'Samples': 206}),
    'eegnet': (EEGNet, {'nb_classes': 2, 'Chans': 6, 'Samples': 206}),
    'cnn1': (CNN1, {}),
    'ucnn1': (UCNN1, {}),
    'cnn3': (CNN3, {}),
    'ucnn3': (UCNN3, {}),
    'cnnr': (CNNR, {}),
    'fcnn': (FCNN, {}),
    'oclnn': (OCLNN, {}),
    'sepconv1d32': (SepConv1D, {'Chans': 6, 'Samples': 206, 'Filters': 32}),
    'sepconv1d16': (SepConv1D, {'Chans': 6, 'Samples': 206, 'Filters': 16}),
    'sepconv1d8': (SepConv1D, {'Chans': 6, 'Samples': 206, 'Filters': 8})
}

def main():
    """
    Main function
    """
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
            description="Evaluates single-trial cross-subject P300 detection using cross-validation")
        parser.add_argument("name", type=str,
                            help="Name of the architecture")
        args = parser.parse_args()

        try:
            model, params = names[args.name.lower()]
            with open(args.name + '.txt','w') as f:                
                model(**params).summary(print_fn = lambda s: f.write(s + '\n'))
        except KeyError:
            print('Unknown architecture {0}! Choose one of the following: {1}'.format(args.name, names.keys()))
            
    except SystemExit:
        print('for help use --help')
        sys.exit(2)
        
if __name__ == "__main__":
    main()
