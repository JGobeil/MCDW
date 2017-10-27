#!/bin/bash

pwd=$(pwd)

for dir in run1 run2 run3 run4 run5
do
    cd $dir
    python mcdw.py | tee run.log
    cd $pwd
done


