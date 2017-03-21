#!/bin/bash

for i in $(seq 1 $1); do
  echo "$i" "${@:2}"
  #python sample.py ${@:2} 2> /dev/null
  sbatch sample-slurm.sh -r run$i ${@:2} 2> /dev/null > /dev/null
done
