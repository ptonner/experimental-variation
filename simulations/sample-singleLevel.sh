
for p in `seq 6 4 14`; do
  for i in `seq .8 .2 .8`; do
    for j in `seq 0 .2 .8`; do
        echo $p, $i, $j;
        for n in {1..20}; do
          #sbatch --array=0-2%1 sample-slurm-array.sh _withHierarchy --chain -l ds$n -n 20000 -t 10 -b 0 single-level/$p-$i-$j > /dev/null 2> /dev/null &
          #sbatch --array=0-2%1 sample-slurm-array.sh _noHierarchy   --chain -l ds$n -n 20000 -t 10 -b 0 single-level/$p-$i-$j > /dev/null $
        done
    done
  done
done
