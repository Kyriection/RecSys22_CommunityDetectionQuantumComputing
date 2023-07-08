#!/bin/bash

dataset=$1
thread=$2
[ -e /tmp/fd1 ]|| mkfifo /tmp/fd1
exec 5<>/tmp/fd1
rm -rf /tmp/fd1 
for i in `seq 1 $thread`
do
	echo ''>&5
done

mkdir -p logs/
mkdir -p results/

# ------------- Each Item ---------------
read -u5
{
  echo Each Item
  time python CT_recommendation.py -d $dataset --EI -o results/EI > logs/ctr-EI.log 2>&1
  echo ''>&5
}&
# -------------- Quantity ---------------
method_list=(LTBipartiteProjectedCommunityDetection)
# T_list=(1 3 5)
T_list=(1)
# alpha_list=(0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0)
alpha_list=(0.0001)
for method in ${method_list[*]}
do
  for T in ${T_list[*]}
  do
    for alpha in ${alpha_list[*]}
    do
    read -u5
    {
      tag=Quantity-$method-$T-$alpha
      echo $tag
      time python CT_community_detection.py $method -d $dataset -a $alpha -t $T -o results/$tag > logs/ctcd-$tag.log 2>&1
      time python CT_qa_recommendation.py   $method -d $dataset -a $alpha -t $T -o results/$tag > logs/ctqr-$tag.log 2>&1
      echo ''>&5
    }&
    done
  done
done

#--------thead---------
wait
echo "total time consume is $SECONDS"
exec 5<&-
exec 5>&-