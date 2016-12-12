#!/bin/bash

for j in `seq 6 4 14`; do
  for i in single-level/$j-0.8-[0-9]*; do
    echo $i
    sbatch analysis-slurm.sh $i 0 > /dev/null 2> /dev/null &
    #sbatch analysis-slurm.sh $i 200 > /dev/null 2> /dev/null
    #sbatch analysis-slurm.sh $i 30 > /dev/null 2> /dev/null
  done
done
