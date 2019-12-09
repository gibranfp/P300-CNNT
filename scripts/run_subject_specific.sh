for i in `seq 0 $1`;
do
    python3 $2 $3 $4 $5 --subject $i
done
