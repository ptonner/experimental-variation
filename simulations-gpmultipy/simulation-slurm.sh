#!/bin/bash

cd /dscrhome/pt59/dev/experimental-variation
source bin/activate
cd simulations

echo $@
$@
