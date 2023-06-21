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
mkdir -p results/

# ------------- Each Item ---------------
read -u5
{
  echo Each Item
  time python CT_recommendation.py --EI -o results/EI > logs/ctr-EI.log 2>&1
  echo ''>&5
}&
# ------------- Total Clusetring ----------
# ------------- Classical ----------
read -u5
{
  echo Kmeans Total w/o attribute
  time python CT_recommendation.py -o results/Kmeans-T > logs/ctr-Kmeans-T.log 2>&1
  echo ''>&5
}&
read -u5
{
  echo Kmeans Total w/ attribute
  time python CT_recommendation.py --attribute -o results/Kmeans-T-A > logs/ctr-Kmeans-T-A.log 2>&1
  echo ''>&5
}&
read -u5
{
  echo Kmeans Bipartite w/o attribute
  time python CT_community_detection.py KmeansCommunityDetection -o results/Kmeans-B > logs/ctcd-Kmeans-B.log 2>&1
  time python CT_qa_recommendation.py KmeansCommunityDetection -o results/Kmeans-B > logs/ctqr-Kmeans-B.log 2>&1
  echo ''>&5
}&
read -u5
{
  echo Kmeans Bipartite w/ attribute
  time python CT_community_detection.py KmeansCommunityDetection --attribute -o results/Kmeans-B-A > logs/ctcd-Kmeans-B-A.log 2>&1
  time python CT_qa_recommendation.py KmeansCommunityDetection --attribute -o results/Kmeans-B-A > logs/ctqr-Kmeans-B-A.log 2>&1
  echo ''>&5
}&
# ------------- Quantum -----------------
read -u5
{
  echo QUBOBipartiteCommunityDetection
  time python CT_community_detection.py QUBOBipartiteCommunityDetection -o results/BM > logs/ctcd-BM.log 2>&1
  time python CT_qa_recommendation.py QUBOBipartiteCommunityDetection -o results/BM > logs/ctqr-BM.log 2>&1
  echo ''>&5
}&
read -u5
{
  echo QUBOBipartiteProjectedCommunityDetection
  time python CT_community_detection.py QUBOBipartiteProjectedCommunityDetection -o results/WPM > logs/ctcd-WPM.log 2>&1
  time python CT_qa_recommendation.py QUBOBipartiteProjectedCommunityDetection -o results/WPM > logs/ctqr-WPM.log 2>&1
  echo ''>&5
}&

# ------------- Our method ----------
# ------------- Cascade -------------
# beta_list=(1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625)
# method_list=(QUBOBipartiteCommunityDetection QUBOBipartiteProjectedCommunityDetection)
beta_list=(0.03125)
method_list=(QUBOBipartiteCommunityDetection)
for method in method_list
do
  for beta in ${beta_list[*]}
  do
  read -u5
  {
    tag=Cascade-$method-$beta
    echo $tag
    time python CT_community_detection.py $method --attribute -b $beta -o results/$tag > logs/ctcd-$tag.log 2>&1
    time python CT_qa_recommendation.py $method --attribute -b $beta -o results/$tag > logs/ctqr-$tag.log 2>&1
    echo ''>&5
  }&
  done
done
# -------------- Quantity ---------------
# method_list=(LTBipartiteCommunityDetection LTBipartiteProjectedCommunityDetection)
# T_list=(1 3 5 7 9)
# alpha_list=(1.0 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0)
method_list=(LTBipartiteCommunityDetection)
T_list=(3)
alpha_list=(0.001)
for method in method_list
do
  for T in ${T_list[*]}
  do
    for alpha in ${alpha_list[*]}
    do
    read -u5
    {
      tag=Quantity-$method-$T-$beta
      echo $tag
      time python CT_community_detection.py $method -a $alpha -t $T -o results/$tag > logs/ctcd-$tag.log 2>&1
      time python CT_qa_recommendation.py $method -a $alpha -t $T -o results/$tag > logs/ctqr-$tag.log 2>&1
      echo ''>&5
    }&
    done
  done
done

# -----------clusterd tail-----------
# cut_list=(0.000 0.025 0.050 0.075 0.100 0.125 0.150)
# for cut in ${cut_list[*]}
# do
# read -u5
# {
#   echo cut_ratio=$cut%
#   echo 'start community detection'
#   time python CT_community_detection.py -c $cut -o results-$cut > logs/ctcd-$cut.log 2>&1
#   echo 'start recommendation'
#   time python CT_qa_recommendation.py -c $cut -o results-$cut > logs/ctqr-$cut.log 2>&1
#   echo ''>&5
# }&
# done

#--------thead---------
wait
echo "total time consume is $SECONDS"
exec 5<&-
exec 5>&-