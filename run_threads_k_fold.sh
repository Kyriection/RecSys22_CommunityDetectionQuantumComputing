#!/bin/bash

dataset=$1
thread=$2
n_folds=$3
[ -e /tmp/fd1 ]|| mkfifo /tmp/fd1
exec 5<>/tmp/fd1
rm -rf /tmp/fd1 
for i in `seq 1 $thread`
do
	echo ''>&5
done

mkdir -p logs/
mkdir -p results/


# ------------- Quantum -----------------
read -u5
{
  method=QUBOBipartiteProjectedCommunityDetection
  echo $method
  time python kfold_LT_community_detection.py $method -d $dataset -k $n_folds -o results/WPM > logs/ctcd-WPM.log 2>&1
  time python kfold_LT_qa_recommendation.py   $method -d $dataset -k $n_folds -o results/WPM > logs/ctqr-WPM.log 2>&1
  echo ''>&5
}&
read -u5
{
  method=QUBOBipartiteProjectedCommunityDetection
  echo $method-implicit
  time python kfold_LT_community_detection.py $method -d $dataset -k $n_folds -o results/WPM-i --implicit > logs/ctcd-WPM-i.log 2>&1
  time python kfold_LT_qa_recommendation.py   $method -d $dataset -k $n_folds -o results/WPM-i --implicit > logs/ctqr-WPM-i.log 2>&1
  echo ''>&5
}&

# ------------- Our method ----------
# ------------- Cascade -------------
beta_list=(0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625)
# beta_list=(0.5 0.25 0.125 0.0625 0.03125 0.015625)
method_list=(QUBOBipartiteProjectedCommunityDetection)
for method in ${method_list[*]}
do
  for beta in ${beta_list[*]}
  do
  read -u5
  {
    tag=Cascade-$method-$beta
    echo $tag
    time python kfold_LT_community_detection.py $method -d $dataset -k $n_folds --attribute -b $beta -o results/$tag > logs/ctcd-$tag.log 2>&1
    time python kfold_LT_qa_recommendation.py   $method -d $dataset -k $n_folds --attribute -b $beta -o results/$tag > logs/ctqr-$tag.log 2>&1
    echo ''>&5
  }&
  read -u5
  {
    tag=Cascade-$method-$beta-implicit
    echo $tag
    time python kfold_LT_community_detection.py $method -d $dataset -k $n_folds --attribute -b $beta -o results/$tag --implicit > logs/ctcd-$tag.log 2>&1
    time python kfold_LT_qa_recommendation.py   $method -d $dataset -k $n_folds --attribute -b $beta -o results/$tag --implicit > logs/ctqr-$tag.log 2>&1
  }&
  done
done
# -------------- Quantity ---------------
method_list=(LTBipartiteProjectedCommunityDetection)
T_list=(1 3 5)
# alpha_list=(0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0)
alpha_list=(0.1 0.01 0.001 0.0001 0)
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
      time python kfold_LT_community_detection.py $method -d $dataset -k $n_folds -a $alpha -t $T -o results/$tag > logs/ctcd-$tag.log 2>&1
      time python kfold_LT_qa_recommendation.py   $method -d $dataset -k $n_folds -a $alpha -t $T -o results/$tag > logs/ctqr-$tag.log 2>&1
      echo ''>&5
    }&
    read -u5
    {
      tag=Quantity-$method-$T-$alpha-implicit
      echo $tag
      time python kfold_LT_community_detection.py $method -d $dataset -k $n_folds -a $alpha -t $T -o results/$tag --implicit > logs/ctcd-$tag.log 2>&1
      time python kfold_LT_qa_recommendation.py   $method -d $dataset -k $n_folds -a $alpha -t $T -o results/$tag --implicit > logs/ctqr-$tag.log 2>&1
    }
    done
  done
done
# ------------- Each Item ---------------
read -u5
{
  echo Each Item
  time python kfold_LT_qa_recommendation.py EachItem -d $dataset -k $n_folds --EI -o results/EI > logs/ctr-EI.log 2>&1
  echo ''>&5
}&
# ------------- Total Clusetring ----------
# ------------- Classical ----------
read -u5
{
  echo Kmeans Total w/o attribute
  time python kfold_LT_qa_recommendation.py KmeansCommunityDetection -d $dataset -k $n_folds -o results/Kmeans-T > logs/ctqr-Kmeans-T.log 2>&1
  echo ''>&5
}&
read -u5
{
  echo Kmeans Total w/ attribute
  time python kfold_LT_qa_recommendation.py KmeansCommunityDetection -d $dataset -k $n_folds --attribute -o results/Kmeans-T-A > logs/ctqr-Kmeans-T-A.log 2>&1
  echo ''>&5
}&
read -u5
{
  echo Kmeans Bipartite w/o attribute
  time python kfold_LT_community_detection.py KmeansBipartiteCommunityDetection -d $dataset -k $n_folds -o results/Kmeans-B > logs/ctcd-Kmeans-B.log 2>&1
  time python kfold_LT_qa_recommendation.py   KmeansBipartiteCommunityDetection -d $dataset -k $n_folds -o results/Kmeans-B > logs/ctqr-Kmeans-B.log 2>&1
  echo ''>&5
}&
read -u5
{
  echo Kmeans Bipartite w/ attribute
  time python kfold_LT_community_detection.py KmeansBipartiteCommunityDetection -d $dataset -k $n_folds --attribute -o results/Kmeans-B-A > logs/ctcd-Kmeans-B-A.log 2>&1
  time python kfold_LT_qa_recommendation.py   KmeansBipartiteCommunityDetection -d $dataset -k $n_folds --attribute -o results/Kmeans-B-A > logs/ctqr-Kmeans-B-A.log 2>&1
  echo ''>&5
}&


#--------thead---------
wait
echo "total time consume is $SECONDS"
exec 5<&-
exec 5>&-