#!/bin/bash
timestamps="rgb_timestamps.txt"
temp="tmp.txt"
groundtruth="rgbd_dataset_freiburg1_desk-groundtruth.txt"
num_samples=`cat $1 | wc -l`

cat $timestamps | head -$num_samples > $temp
paste $temp $1 > $1.result
python evaluate_ate.py $groundtruth $1.result --plot $1.png --offset 0 --scale 1 --verbose


