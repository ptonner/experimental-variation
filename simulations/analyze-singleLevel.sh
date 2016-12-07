#!/bin/bash

for i in single-level/5-[0-9]*; do
  echo $i
  python analysis.py -p $i -b 200
done
