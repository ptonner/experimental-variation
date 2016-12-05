
for p in `seq 5 5 30`; do
  for i in `seq .2 .2 .8`; do
    for j in `seq 0 .2 .8`; do
        echo $p, $i, $j;
        for n in {1..10}; do
          # echo "python generate.py -l ds$n single-level/$p-$i-$j"
          #python sample.py -l ds$n -r run$SLURM_ARRAY_TASK_ID single-level/$p-$i-$j 2> /dev/null;

          sbatch --array=0-10%1 sample-slurm-batch.sh single-level/$p-$i-$j  ds$n run$SLURM_ARRAY_TASK_ID -1
          # sbatch --array=0-10%1 sample-slurm-batch.sh single-level/$p-$i-$j  ds$n run$SLURM_ARRAY_TASK_ID -1
        done
    done
  done
done
