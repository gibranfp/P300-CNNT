mkdir -p $3/SepConv1D_{64,32,16,8,4,2,1}F
for i in 1 2 4 8 16 32 64;
do
    python3 src/cross_subject_SepConv1D.py $1 $2 $3/SepConv1D_$iF/ --n_filters $i
done
