#!/bin/bash

# cd /dscrhome/pt59/dev/experimental-variation
# source bin/activate
# cd simulations

for p in `seq 6 4 14`; do
  # for i in `seq .8 .2 .8`; do
    for j in `seq 0 .2 .8`; do
        # echo $p, $i, $j;
        # python generate-singleLevel.py $p $i $j;

        echo $p, $j;
        python generate-singleLevel.py $p 2 .9 $j;

        for n in {1..20}; do
          # echo $n
          # python generate.py -l ds$n single-level/$p-$i-$j
          python generate.py -l ds$n single-level/$p-2.0-0.9-$j 2> /dev/null

        done

        python analysis.py -p single-level/$p-2.0-0.9-$j 2> /dev/null
    done
  # done
done
