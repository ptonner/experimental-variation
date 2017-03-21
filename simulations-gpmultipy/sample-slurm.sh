#!/bin/bash
#SBATCH --job-name=gp-simulation
#SBATCH --output=output/gp-simulation.%j.out
#SBATCH --mem=4000

cd /dscrhome/pt59/dev/experimental-variation
source bin/activate
cd simulations
#python sample.py -n 10000 -t 10 -b 200 -l ds$2 -r run$3 -v $4 ${@:5} $1 2> /dev/null
python sample.py $@ 2> /dev/null
