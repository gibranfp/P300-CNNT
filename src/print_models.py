from CNN1 import UCNN1, CNN1
from EEGModels import ShallowConvNet, DeepConvNet, EEGNet
from BN3model import BN3
from FCNNmodel import FCNN

            
bn3 = BN3()
with open('BN3.txt','w') as fh:
    bn3.summary(print_fn = lambda x: fh.write(x + '\n'))
        
cnn1 = CNN1()
with open('CNN1.txt','w') as fh:
    cnn1.summary(print_fn = lambda x: fh.write(x + '\n'))

ucnn1 = UCNN1()
with open('UCNN1.txt','w') as fh:
    ucnn1.summary(print_fn = lambda x: fh.write(x + '\n'))

scn = ShallowConvNet(2)
with open('ShallowConvNet.txt','w') as fh:
    scn.summary(print_fn = lambda x: fh.write(x + '\n'))

dcn = DeepConvNet(2)
with open('DeepConvNet.txt','w') as fh:
    dcn.summary(print_fn = lambda x: fh.write(x + '\n'))

egn = EEGNet(2)
with open('EEGNet.txt','w') as fh:
    egn.summary(print_fn = lambda x: fh.write(x + '\n'))

fcnn = FCNN()
with open('FCNN.txt','w') as fh:
    fcnn.summary(print_fn = lambda x: fh.write(x + '\n'))
