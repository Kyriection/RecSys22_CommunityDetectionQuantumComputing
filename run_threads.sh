#!/bin/bash

thread=$1
[ -e /tmp/fd1 ]|| mkfifo /tmp/fd1
exec 5<>/tmp/fd1
rm -rf /tmp/fd1 
for i in `seq 1 $thread`
do
	echo ''>&5
done

mkdir -p logs/

# -----------clusterd tail-----------
cut_list=(0.000 0.025 0.050 0.075 0.100 0.125 0.150)
for cut in ${cut_list[*]}
do
read -u5
{
  echo cut_ratio=$cut%
  echo 'start community detection'
  time python CT_community_detection.py -c $cut -o results-$cut > logs/ctcd-$cut.log 2>&1
  echo 'start recommendation'
  time python CT_qa_recommendation.py -c $cut -o results-$cut > logs/ctqr-$cut.log 2>&1
  echo ''>&5
}&
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
  read -u5
  {
    echo T=$T, alpha=$alpha
    echo 'start community detection'
    time python CT_community_detection.py -t $T -a $alpha -l 100 -o results-$T-$alpha > logs/ctcd-$T-$alpha.log 2>&1
    echo 'start recommendation'
    time python CT_qa_recommendation.py -t $T -a $alpha -l 100 -o results-$T-$alpha > logs/ctqr-$T-$alpha.log 2>&1
    echo ''>&5
  }
  done
done

#--------thead---------
wait
echo "total time consume is $SECONDS"
exec 5<&-
exec 5>&-