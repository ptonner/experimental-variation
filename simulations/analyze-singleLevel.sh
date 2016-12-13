#!/bin/bash

for j in `seq 6 4 14`; do
  for i in single-level/$j-2.0-0.9-[0-9]*; do
    echo $i
    sbatch --output='output/analysis.%j.out' --mem=1000 -J analysis simulation-slurm.sh python analysis.py -cl -b 200 -a withHierarchy $i
    #sbatch analysis-slurm.sh $i 0 > /dev/null 2> /dev/null &
    #sbatch analysis-slurm.sh $i 200 > /dev/null 2> /dev/null
    #sbatch analysis-slurm.sh $i 30 > /dev/null 2> /dev/null
  done
done
