#!/bin/bash

for i in {1..10}; do
 python single-level.py -d results5 -l $i &
 done
