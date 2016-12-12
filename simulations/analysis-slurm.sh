#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --output=output/analysis.%A_%a.out
#SBATCH --error=output/analysis.%A_%a.out
#SBATCH --mem=1000

cd /dscrhome/pt59/dev/experimental-variation
source bin/activate
cd simulations

echo $1

python analysis.py -cpl -b$2 $1
