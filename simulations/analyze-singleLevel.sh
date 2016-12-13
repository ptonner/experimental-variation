#!/bin/bash

for j in `seq 6 4 14`; do
  for i in single-level/$j-2.0-0.9-[0-9]*; do
    echo $i
    #sbatch --output='output/analysis.%j.out' --mem=1000 -J analysis simulation-slurm.sh python analysis.py -cl -b 200 -a withHierarchy $i
    sbatch --output='output/analysis.%j.out' --mem=1000 -J analysis simulation-slurm.sh python analysis.py -cl -b 200 -a noHierarchy $i
  done
done
