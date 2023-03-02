#!/bin/bash
alpha_list=(1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625)
alpha_list=(0.015625)
for alpha in ${alpha_list[*]}
do
  echo alpha=$alpha
  echo 'start run_community_detection_mod'
  python run_community_detection_mod.py -a $alpha > cd.log 2>&1
  echo 'start cd_recommendation'
  python cd_recommendation.py -a $alpha > cdr.log 2>&1
done