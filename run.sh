#!/bin/bash
# alpha_list=(1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625)
# alpha_list=(0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125)
# beta_list=(0.25 0.125 0.0625 0.03125 0.015625 0.0078125)

# alpha_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# alpha_list=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
# alpha_list=(0.25 0.0625 0.015625)
# alpha_list=(0.000 0.025 0.050 0.075 0.100 0.125 0.150)
# alpha_list=(1.00 0.75 0.50 0.25 0.00)
# alpha_list=(1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625)
# alpha_list=(0.05 0.02 0.01 0.0005 0.0001)
# alpha_list=(0 1 2 3 4 5 6)

# for alpha in ${alpha_list[*]}
# do
#   echo alpha=$alpha
#   echo 'start community detection'
#   # time python run_community_detection_mod.py -a $alpha > cd.log 2>&1
#   time python CT_community_detection.py -a 0.005 -t 1 -l $alpha > ctcd.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
#   echo 'start recommendation'
#   time python CT_qa_recommendation.py -a 0.005 -t 1 -l $alpha > ctqr.log 2>&1
#   # time python cd_recommendation.py -a $alpha > cdr.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
#   # echo 'start non-quantum'
#   # time python CT_recommendation.py -c $alpha > ctr.log 2>&1
# done

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

# -----------clusterd tail-----------
cut_list=(0.000 0.025 0.050 0.075 0.100 0.125 0.150)
for cut in ${cut_list[*]}
do
  echo cut_ratio=$cut%
  echo 'start community detection'
  time python CT_community_detection.py -c $cut > ctcd.log 2>&1
  if [ $? -ne 0 ]; then
    echo 'error'
    break
  fi
  echo 'start recommendation'
  time python CT_qa_recommendation.py -c $cut > ctqr.log 2>&1
  if [ $? -ne 0 ]; then
    echo 'error'
    break
  fi
done

# -----------layer of A1-------------
# layer_list=(0 1 2 3 4 5 6 7 8 9 10)
# for layer in ${layer_list[*]}
# do
#   echo layer=$layer
#   echo 'start community detection'
#   time python CT_community_detection.py -l $layer > ctcd.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
#   echo 'start recommendation'
#   time python CT_qa_recommendation.py -l $layer > ctqr.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
# done

# -----------our method---------------
T_list=(1 3 5 7 9)
alpha_list=(1.0 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001 0)
for T in ${T_list[*]}
do
  for alpha in ${alpha_list[*]}
  do
    echo T=$T, alpha=$alpha
    echo 'start community detection'
    time python CT_community_detection.py -t $T -a $alpha > ctcd.log 2>&1
    if [ $? -ne 0 ]; then
      echo 'error'
      break
    fi
    echo 'start recommendation'
    time python CT_qa_recommendation.py -t $T -a $alpha > ctqr.log 2>&1
    if [ $? -ne 0 ]; then
      echo 'error'
      break
    fi
  done
done