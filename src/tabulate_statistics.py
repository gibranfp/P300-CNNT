import pandas as pd
import glob
import sys 
import os

f2n = {
    'bn3':'BN3',
    'shallowconvnet':'ShallowConvNet',
    'deepconvnet':'DeepConvNet',
    'eegnet':'EEGNet',
    'cnn1':'CNN1',
    'ucnn1':'UCNN1',
    'cnn3':'CNN3',
    'ucnn3':'UCNN3',
    'cnnr':'CNNR',
    'fcnn':'FCNN',
    'oclnn':'OCLNN',
    'sepconv1d_32f':'SepConv1D-32F',
    'sepconv1d_16f':'SepConv1D-16F',
    'sepconv1d_8f':'SepConv1D-8F',
    'sepconv1d_4f':'SepConv1D-4F',
    'sepconv1d_2f':'SepConv1D-2F',
    'sepconv1d_1f':'SepConv1D-1F',
    'sepconv1dext_32f':'SepConv1DExt-32F',
    'sepconv1dext_16f':'SepConv1DExt-16F',
    'sepconv1dext_8f':'SepConv1DExt-8F',
    'sepconv1dext_4f':'SepConv1DExt-4F',
    'sepconv1dext_2f':'SepConv1DExt-2F',
    'sepconv1dext_1f':'SepConv1DExt-1F'
}

d2n = {
    'uam':'$mathbb[D]_1$',
    'horizon':'$mathbb[D]_4$',
    'bci_ii':'$mathbb[D]_2$',
    'bci_iii':'$mathbb[D]_3$'
}

ds = os.listdir(sys.argv[1])
datasets = []
arqs = []
li = []
for d in ds:
    all_csv = glob.glob(sys.argv[1] + '/' + d + "/*_all.csv")
    for c in all_csv:
        df = pd.read_csv(c, index_col=None, header=0)
        li.append(df)
        datasets.append(d2n[d])
        arq_name = f2n[os.path.basename(os.path.splitext(c)[0]).replace("_all", "")]
        arqs.append(arq_name)

frame = pd.concat(li, axis=0, ignore_index=True)
frame['Dataset'] = datasets
frame['Architecture'] = arqs

# Params,Trainable,NonTrainable,FLOPS,Mean Epoch Time,Total Train Time,Epochs,Train Size,Valid Size,Test Time,Test Size,Test per example
frame = frame[['Dataset','Architecture','#Params', 'FLOPS', 'Test per example']]
print(frame.to_latex())
