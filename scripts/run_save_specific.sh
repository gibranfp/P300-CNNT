for f in 1 2 4 8;
do
    echo "$f filters"
    for i in `seq 0 21`;
    do
        python3 src/save_specific_SepConv1D.py $1 $2 $3/$f/ --subject $i --n_filters $f
    done
done
