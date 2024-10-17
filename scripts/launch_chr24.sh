# setup
python3 chr24_prepare.py
if [ ! -d "../results" ]; then
    mkdir ../results
fi
if [ ! -d "../results/individual_results" ]; then
    mkdir ../results/individual_results
fi

# compute likelihoods
for P0 in 0 1
do
    cp "bookprob${P0}.cpp" "../InnovationProcessesInference/src/bookprob.cpp"
    DNOW=$(pwd)
    cd ../InnovationProcessesInference/
    make -j
    cd $DNOW
    pids=()
    for chunk in {0..7}
    do
        nohup python3 chr24_compute.py ${chunk} ${P0} 8 &
        pids[${chunk}]=$!
    done

    for pid in ${pids[*]}; do
        wait $pid
    done
done

# summarise likelihoods
if [ ! -d "../results/joined_results" ]; then
    mkdir ../results/joined_results
fi
python3 chr24_join.py

python3 chr24_results.py