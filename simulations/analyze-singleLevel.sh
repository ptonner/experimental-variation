#!/bin/bash

for i in single-level/[0-9]*; do
  echo $i
  #python analysis.py -p $i -b 200
  sbatch analysis-slurm.sh -cb 200 $i > /dev/null 2> /dev/null
done
