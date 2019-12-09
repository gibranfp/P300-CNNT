mkdir -p models/horizon/cross_subject/{BN3,EEGNet,ShallowConvNet,DeepConvNet,FCNN,OCLNN,CNN1,UCNN1,CNN3,UCNN3,CNNR,SepConv1D}
python3 src/cross_subject_BN3.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/BN3/
python3 src/cross_subject_OCLNN.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/OCLNN/
python3 src/cross_subject_CNN1.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/CNN1/
python3 src/cross_subject_CNN3.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/CNN3/
python3 src/cross_subject_CNNR.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/CNNR/
python3 src/cross_subject_UCNN1.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/UCNN1/
python3 src/cross_subject_UCNN3.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/UCNN3/
python3 src/cross_subject_EEGNet.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/EEGNet/
python3 src/cross_subject_ShallowConv1D.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/ShallowConvNet/
python3 src/cross_subject_DeepConvNet.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/DeepConvNet/
python3 src/cross_subject_SepConv1D.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/SepConv1D/
python3 src/cross_subject_FCNN.py data/data_horizon_8ch.npy data/labels_horizon_8ch.npy models/horizon/cross_subject/FCNN/
