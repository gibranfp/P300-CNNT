mkdir -p models/uam/subject_specific/{BN3,EEGNet,ShallowConvNet,DeepConvNet,FCNN,OCLNN,CNN1,UCNN1,CNN3,UCNN3,CNNR,SepConv1D}
bash scripts/run_subject_specific.sh 21 src/subject_specific_BN3.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/BN3/
bash scripts/run_subject_specific.sh 21 src/subject_specific_EEGNet.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/EEGNet/
bash scripts/run_subject_specific.sh 21 src/subject_specific_ShallowConvNet.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/ShallowConvNet/
bash scripts/run_subject_specific.sh 21 src/subject_specific_DeepConvNet.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/DeepConvNet/
bash scripts/run_subject_specific.sh 21 src/subject_specific_FCNN.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/FCNN/
bash scripts/run_subject_specific.sh 21 src/subject_specific_OCLNN.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/OCLNN/
bash scripts/run_subject_specific.sh 21 src/subject_specific_CNN1.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/CNN1/
bash scripts/run_subject_specific.sh 21 src/subject_specific_UCNN1.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/UCNN1/
bash scripts/run_subject_specific.sh 21 src/subject_specific_CNN3.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/CNN3/
bash scripts/run_subject_specific.sh 21 src/subject_specific_UCNN3.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/UCNN3/
bash scripts/run_subject_specific.sh 21 src/subject_specific_CNNR.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/CNNR/
bash scripts/run_subject_specific.sh 21 src/subject_specific_SepConv1D.py data/data_uam_6ch.npy data/labels_uam_6ch.npy models/uam/subject_specific/SepConv1D/
