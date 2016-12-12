#!/bin/bash
#SBATCH --job-name=gp-simulation
#SBATCH --output=output/gp-simulation.%A_%a.out
#SBATCH --error=output/gp-simulation.%A_%a.out
#SBATCH --mem=1000

cd /dscrhome/pt59/dev/experimental-variation
source bin/activate
cd simulations

echo $@

python sample.py -r run$SLURM_ARRAY_TASK_ID$1 ${@:2}
