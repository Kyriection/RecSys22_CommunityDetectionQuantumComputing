#!/bin/bash
# alpha_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
alpha_list=(1.0)
for alpha in ${alpha_list[*]}
do
  echo alpha=$alpha
  echo 'start run_community_detection_mod'
  python run_community_detection_mod.py $alpha > cd.log 2>&1
  echo 'start cd_recommendation'
  python cd_recommendation.py $alpha > cdr.log 2>&1
done