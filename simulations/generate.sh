#!/bin/bash

printf "%s\n" "generating $2 datasets from configuration in $1"
for i in $(seq 1 $2); do
 #echo $i;
 printf "%s" "."
 python generate.py -l ds$i $1 2> /dev/null
 done
printf "done!\n"
