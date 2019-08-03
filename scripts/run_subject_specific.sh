for i in `seq 0 21`;
do
    python3 $1 $2 $3 $4 --subject $i
done
