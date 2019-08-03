mkdir -p models/subject_dependent/{BN3,EEGNet,ShallowConvNet,DeepConvNet,FCNN,OCLNN,CNN1,UCNN1,CNN3,UCNN3,CNNR,SepConv1D}
bash scripts/run_subject_dependent.sh src/single_trial_BN3.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/BN3/
bash scripts/run_subject_dependent.sh src/single_trial_EEGNet.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/EEGNet/
bash scripts/run_subject_dependent.sh src/single_trial_ShallowConvNet.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/ShallowConvNet/
bash scripts/run_subject_dependent.sh src/single_trial_DeepConvNet.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/DeepConvNet/
bash scripts/run_subject_dependent.sh src/single_trial_FCNN.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/FCNN/
bash scripts/run_subject_dependent.sh src/single_trial_OCLNN.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/OCLNN/
bash scripts/run_subject_dependent.sh src/single_trial_CNN1.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/CNN1/
bash scripts/run_subject_dependent.sh src/single_trial_UCNN1.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/UCNN1/
bash scripts/run_subject_dependent.sh src/single_trial_CNN3.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/CNN3/
bash scripts/run_subject_dependent.sh src/single_trial_UCNN3.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/UCNN3/
bash scripts/run_subject_dependent.sh src/single_trial_CNNR.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/CNNR/
bash scripts/run_subject_dependent.sh src/single_trial_SepConv1D.py data/data6_osv.npy data/labels6_osv.npy modedls/subject_dependent/SepConv1D/
