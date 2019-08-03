mkdir -p models/subject_specific/filter_number/SepConv1D_{64,32,16,8,4,2,1}F
for i in `seq 0 21`;
do
    python3 src/subject_specific_SepConv1D.py data/data6_osv.npy data/labels6_osv.npy models/subject_specific/filter_number/SepConv1D_64F/ --subject $i --n_filters 64
    python3 src/subject_specific_SepConv1D.py data/data6_osv.npy data/labels6_osv.npy models/subject_specific/filter_number/SepConv1D_32F/ --subject $i --n_filters 32
    python3 src/subject_specific_SepConv1D.py data/data6_osv.npy data/labels6_osv.npy models/subject_specific/filter_number/SepConv1D_16F/ --subject $i --n_filters 16
    python3 src/subject_specific_SepConv1D.py data/data6_osv.npy data/labels6_osv.npy models/subject_specific/filter_number/SepConv1D_8F/ --subject $i --n_filters 8
    python3 src/subject_specific_SepConv1D.py data/data6_osv.npy data/labels6_osv.npy models/subject_specific/filter_number/SepConv1D_4F/ --subject $i --n_filters 4
    python3 src/subject_specific_SepConv1D.py data/data6_osv.npy data/labels6_osv.npy models/subject_specific/filter_number/SepConv1D_2F/ --subject $i --n_filters 2
    python3 src/subject_specific_SepConv1D.py data/data6_osv.npy data/labels6_osv.npy models/subject_specific/filter_number/SepConv1D_1F/ --subject $i --n_filters 1
done
