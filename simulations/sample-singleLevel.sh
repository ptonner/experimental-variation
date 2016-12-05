
for p in `seq 5 5 30`; do
  for i in `seq .2 .2 .8`; do
    for j in `seq 0 .2 .8`; do
        echo $p, $i, $j;
        for n in {1..10}; do
          # echo "python generate.py -l ds$n single-level/$p-$i-$j"
          sbatch --array=0-10%1 python sample.py -l ds$n -r run$SLURM_ARRAY_TASK_ID single-level/$p-$i-$j 2> /dev/null;
        done
    done
  done
done
