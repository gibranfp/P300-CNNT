for f in 1 2 4 8;
do
    echo "$f filters"
    python3 src/save_cross_SepConv1D.py $1 $2 $3/$f/ --n_filters $f
done
