#!/bin/bash
# alpha_list=(1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625)
# alpha_list=(0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125)
# beta_list=(0.25 0.125 0.0625 0.03125 0.015625 0.0078125)

# alpha_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# alpha_list=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
# alpha_list=(0.25 0.0625 0.015625)
# alpha_list=(0.000 0.025 0.050 0.075 0.100 0.125 0.150)
alpha_list=(0.125)

for alpha in ${alpha_list[*]}
do
  echo alpha=$alpha
  echo 'start community detection'
  # time python run_community_detection_mod.py -a $alpha > cd.log 2>&1
  time python CT_community_detection.py -a $alpha > ctcd.log 2>&1
  echo 'start recommendation'
  time python CT_qa_recommendation.py -a $alpha > ctqr.log 2>&1
done

# for alpha in ${alpha_list[*]}
# do
#   for beta in ${beta_list[*]}
#   do
#     if [ $(echo "$alpha > $beta" | bc) -eq 1 ]
#     then
#       echo alpha=$alpha, beta=$beta
#       echo 'start run_community_detection_mod'
#       # echo python run_community_detection_multi_hybrid.py -a $alpha -b $beta
#       python run_community_detection_multi_hybrid.py -a $alpha -b $beta > cd.log 2>&1
#       echo 'start cd_recommendation'
#       # echo python cd_recommendation.py -a $alpha -b $beta
#       python cd_recommendation.py -a $alpha -b $beta > cdr.log 2>&1
#     fi
#   done
# done