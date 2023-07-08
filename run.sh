#!/bin/bash
# alpha_list=(1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625)
# alpha_list=(0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125)
# beta_list=(0.25 0.125 0.0625 0.03125 0.015625 0.0078125)

# alpha_list=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# alpha_list=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)
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
#   time python LT_community_detection.py -a $alpha > ctcd.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
#   echo 'start recommendation'
#   time python LT_qa_recommendation.py -a $alpha > ctqr.log 2>&1
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
# cut_list=(0.000 0.025 0.050 0.075 0.100 0.125 0.150)
# for cut in ${cut_list[*]}
# do
#   echo cut_ratio=$cut%
#   echo 'start community detection'
#   time python LT_community_detection.py -c $cut > ctcd.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
#   echo 'start recommendation'
#   time python LT_qa_recommendation.py -c $cut > ctqr.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
# done

# -----------layer of A1-------------
# layer_list=(0 1 2 3 4 5 6 7 8 9 10)
# for layer in ${layer_list[*]}
# do
#   echo layer=$layer
#   echo 'start community detection'
#   time python LT_community_detection.py -l $layer > ctcd.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
#   echo 'start recommendation'
#   time python LT_qa_recommendation.py -l $layer > ctqr.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
# done

# -----------our method---------------
# T_list=(1 3 5 7 9)
# alpha_list=(1.0 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001 0)
# for T in ${T_list[*]}
# do
#   for alpha in ${alpha_list[*]}
#   do
#     echo T=$T, alpha=$alpha
#     echo 'start community detection'
#     time python LT_community_detection.py -t $T -a $alpha > ctcd.log 2>&1
#     if [ $? -ne 0 ]; then
#       echo 'error'
#       break
#     fi
#     echo 'start recommendation'
#     time python LT_qa_recommendation.py -t $T -a $alpha > ctqr.log 2>&1
#     if [ $? -ne 0 ]; then
#       echo 'error'
#       break
#     fi
#   done
# done

# echo 'HybridCommunityDetection'
# alpha_list=(1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625)
# for alpha in ${alpha_list[*]}
# do
#   echo $alpha
#   time python LT_community_detection.py QUBOBipartiteProjectedCommunityDetection --cascade -b $alpha -o results-CWPM-$alpha > logs/ctcd-CWPM-$alpha.log 2>&1
#   time python LT_qa_recommendation.py QUBOBipartiteProjectedCommunityDetection --cascade -b $alpha -o results-CWPM-$alpha > logs/ctqr-CWPM-$alpha.log 2>&1
# done
# echo 'CascadeCommunityDetection'
# alpha_list=(1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625)
# for alpha in ${alpha_list[*]}
# do
#   echo $alpha
#   time python LT_community_detection.py -m CascadeCommunityDetection -a $alpha -o results-$alpha > logs/ctcd-$alpha.log 2>&1
#   time python LT_qa_recommendation.py -m CascadeCommunityDetection -a $alpha -o results-$alpha > logs/ctqr-$alpha.log 2>&1
# done
# echo 'QUBOBipartiteCommunityDetection'
# time python LT_community_detection.py -m QUBOBipartiteCommunityDetection > ctcd.log 2>&1
# time python LT_qa_recommendation.py -m QUBOBipartiteCommunityDetection > ctqr.log 2>&1
# echo 'QUBOBipartiteProjectedCommunityDetection'
# time python LT_community_detection.py -m QUBOBipartiteProjectedCommunityDetection > ctcd.log 2>&1
# time python LT_qa_recommendation.py -m QUBOBipartiteProjectedCommunityDetection > ctqr.log 2>&1
# echo 'LTBipartiteCommunityDetection'
# time python LT_community_detection.py -m LTBipartiteCommunityDetection > ctcd.log 2>&1
# time python LT_qa_recommendation.py -m LTBipartiteCommunityDetection > ctqr.log 2>&1
# echo 'LTBipartiteProjectedCommunityDetection'
# time python LT_community_detection.py -m LTBipartiteProjectedCommunityDetection > ctcd.log 2>&1
# time python LT_qa_recommendation.py -m LTBipartiteProjectedCommunityDetection > ctqr.log 2>&1
# echo 'QuantityDivision'
# time python LT_community_detection.py -m QuantityDivision -t 7 -o results-Quantity-7 > ctcd.log 2>&1
# time python LT_qa_recommendation.py -m QuantityDivision -t 7 -o results-Quantity-7 > ctqr.log 2>&1
# echo 'KmeansCommunityDetection'
# time python LT_community_detection.py -m KmeansCommunityDetection -o results-Kmeans-B-I > ctcd.log 2>&1
# time python LT_qa_recommendation.py -m KmeansCommunityDetection -o results-Kmeans-B-I > ctqr.log 2>&1

# echo 'adaptive selection'
# time python LT_qa_recommendation.py -m QUBOBipartiteCommunityDetection > logs/ctqr-BM.log 2>&1
# time python LT_qa_recommendation.py -m QUBOBipartiteProjectedCommunityDetection > logs/ctqr-WPM.log 2>&1
# time python LT_qa_recommendation.py -m KmeansCommunityDetection -o results-Kmeans-B-I > logs/ctqr-kmeans-B-I.log 2>&1
# time python LT_qa_recommendation.py -m HybridCommunityDetection -a 0.125 -o results-WPM-0.125 > logs/ctqr-WPM-0.125.log 2>&1
# time python LT_qa_recommendation.py -m LTBipartiteCommunityDetection -t 3 -a 0.001 -o results--3-0.001 > logs/ctqr-BM-3-0.001.log 2>&1

# T_list=(1 3 5 7 9)
# for T in ${T_list[*]}
# do
#   echo T=$T
#   echo 'start community detection'
#   time python LT_community_detection.py -t $T -m QuantityDivision > ctcd.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
#   echo 'start recommendation'
#   time python LT_qa_recommendation.py -t $T -m QuantityDivision > ctqr.log 2>&1
#   if [ $? -ne 0 ]; then
#     echo 'error'
#     break
#   fi
# done

# T_list=(1 3 5 7)
# alpha_list=(1.0 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001 0)
# method_list=(LTBipartiteCommunityDetection LTBipartiteProjectedCommunityDetection)
# for method in ${method_list[*]}
# do
#   for T in ${T_list[*]}
#   do
#     for alpha in ${alpha_list[*]}
#     do
#       tag=$method-$T-$alpha
#       echo $tag
#       echo 'start community detection'
#       time python LT_community_detection.py -m $method -t $T -a $alpha -o results-$tag > logs/ctcd-$tag.log 2>&1
#       if [ $? -ne 0 ]; then
#         echo 'error'
#         break
#       fi
#       echo 'start recommendation'
#       time python LT_qa_recommendation.py -m $method -t $T -a $alpha -o results-$tag > logs/ctqr-$tag.log 2>&1
#       if [ $? -ne 0 ]; then
#         echo 'error'
#         break
#       fi
#     done
#   done
# done

# ------------- Each Item ---------------
# echo Each Item
# time python CT_recommendation.py --EI -o results/EI > logs/ctr-EI.log 2>&1

# echo 'HybridCommunityDetection'
# # beta_list=(1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625)
# beta_list=(0.125)
# for beta in ${beta_list[*]}
# do
#   echo $beta
#   time python LT_community_detection.py QUBOBipartiteCommunityDetection --attribute -b $beta -o results-CBM-$beta > logs/ctcd-CBM-$beta.log 2>&1
#   time python LT_qa_recommendation.py QUBOBipartiteCommunityDetection --attribute -b $beta -o results-CBM-$beta > logs/ctqr-CBM-$beta.log 2>&1
# done
# echo 'UserBipartiteCommunityDetection'
# time python LT_community_detection.py UserBipartiteCommunityDetection > ctcd.log 2>&1
# time python LT_qa_recommendation.py UserBipartiteCommunityDetection > ctqr.log 2>&1

# method=LTBipartiteCommunityDetection
# T=3
# alpha=0.0005
# beta=0.03125
# tag=Q-C-$method-$T-$alpha-$beta
# echo $tag
# time python LT_community_detection.py $method --attribute -b $beta -a $alpha -t $T -o results/$tag > logs/ctcd-$tag.log 2>&1
# time python LT_qa_recommendation.py $method --attribute -b $beta -a $alpha -t $T -o results/$tag > logs/ctqr-$tag.log 2>&1

# method=LTBipartiteProjectedCommunityDetection
# T=5
# alpha=0.0001
# beta=0.03125
# tag=Q-C-$method-$T-$alpha-$beta
# echo $tag
# time python LT_community_detection.py $method --attribute -b $beta -a $alpha -t $T -o results/$tag > logs/ctcd-$tag.log 2>&1
# time python LT_qa_recommendation.py $method --attribute -b $beta -a $alpha -t $T -o results/$tag > logs/ctqr-$tag.log 2>&1

# method=LTBipartiteCommunityDetection
# T=3
# alpha=0.001
# tag=-$T-$alpha
# echo $tag
# time python LT_community_detection.py $method -a $alpha -t $T -o results-$tag > logs/ctcd-$tag.log 2>&1
# time python LT_qa_recommendation.py $method -a $alpha -t $T -o results-$tag > logs/ctqr-$tag.log 2>&1

# method=LTBipartiteProjectedCommunityDetection
# T=5
# alpha=0.0001
# tag=$method-$T-$alpha
# echo $tag
# time python LT_community_detection.py $method -a $alpha -t $T -o results/$tag > logs/ctcd-$tag.log 2>&1
# time python LT_qa_recommendation.py $method -a $alpha -t $T -o results/$tag > logs/ctqr-$tag.log 2>&1


# beta_list=(0.125 0.03125)
# for beta in ${beta_list[*]}
# do
#   echo $beta
#   time python LT_community_detection.py HybridCommunityDetection -b $beta -o results-CBM-$beta > logs/ctcd-CBM-$beta.log 2>&1
#   time python LT_qa_recommendation.py HybridCommunityDetection -b $beta -o results-CBM-$beta > logs/ctqr-CBM-$beta.log 2>&1
# done

# method=WPM
# T=3
# alpha=0.005
# beta=0.125
# tag=Q-C-$method-$T-$alpha-$beta
# echo $tag
# time python LT_community_detection.py HybridCommunityDetection -b $beta -a $alpha -t $T -o results/$tag > logs/ctcd-$tag.log 2>&1
# time python LT_qa_recommendation.py HybridCommunityDetection -b $beta -a $alpha -t $T -o results/$tag > logs/ctqr-$tag.log 2>&1

# method_list=(LTBipartiteProjectedCommunityDetection LTBipartiteCommunityDetection)
# T_list=(3 5)
# alpha_list=(0.005 0.001 0.0005 0.0001)
# beta_list=(0.125 0.0625 0.03125)
# for method in ${method_list[*]}
# do
#   for T in ${T_list[*]}
#   do
#     for alpha in ${alpha_list[*]}
#     do
#       for beta in ${beta_list[*]}
#       do
#         tag=Q-C-$method-$T-$alpha-$beta
#         echo $tag
#         time python LT_community_detection.py $method --attribute -b $beta -a $alpha -t $T -o results/$tag > logs/ctcd-$tag.log 2>&1
#         time python LT_qa_recommendation.py   $method --attribute -b $beta -a $alpha -t $T -o results/$tag > logs/ctqr-$tag.log 2>&1
#       done
#     done
#   done
# done

# beta_list=(0.65)
# for beta in ${beta_list[*]}
# do
#   tag=test-$beta
#   echo $tag
#   time python LT_community_detection.py TestCommunityDetection -b $beta -o results/$tag > logs/ctcd-$tag.log 2>&1
#   time python LT_qa_recommendation.py   TestCommunityDetection -b $beta -o results/$tag > logs/ctqr-$tag.log 2>&1
# done

# time python LT_qa_recommendation.py QUBOBipartiteCommunityDetection -o results/BM > logs/ctqr-BM.log 2>&1
# time python LT_qa_recommendation.py QUBOBipartiteProjectedCommunityDetection -o results/WPM > logs/ctqr-WPM.log 2>&1
# method=QUBOBipartiteProjectedCommunityDetection
# beta=0.03125
# tag=CWPM-$beta
# time python LT_qa_recommendation.py $method --attribute -b $beta -o results-$tag > logs/ctqr-$tag.log 2>&1

echo QUBOBipartiteProjectedCommunityDetection
# time python LT_community_detection.py QUBOBipartiteProjectedCommunityDetection -d MovielensSample3 -o results/WPM > logs/ctcd-WPM.log 2>&1
# time python LT_qa_recommendation.py   QUBOBipartiteProjectedCommunityDetection -d MovielensSample3 -o results/WPM > logs/ctqr-WPM.log 2>&1
time python LT_qa_run_community_detection.py QUBOBipartiteProjectedCommunityDetection -d MovielensSample3 -o results/WPM > logs/ltqcd-WPM.log 2>&1
# time python LT_qa_recommendation.py   QUBOBipartiteProjectedCommunityDetection -d MovielensSample3 -o results/WPM > logs/ctqr-WPM.log 2>&1


# ------------- Cascade -------------
# beta_list=(0.0625)
# method_list=(QUBOBipartiteProjectedCommunityDetection)
# for method in ${method_list[*]}
# do
#   for beta in ${beta_list[*]}
#   do
#     tag=Cascade-$method-$beta
#     echo $tag
#     time python LT_community_detection.py $method --attribute -b $beta -o results/$tag > logs/ctcd-$tag.log 2>&1
#     time python LT_qa_recommendation.py $method --attribute -b $beta -o results/$tag > logs/ctqr-$tag.log 2>&1
#   done
# done

# echo Kmeans Bipartite w/o attribute
# time python LT_community_detection.py KmeansCommunityDetection -o results/Kmeans-B > logs/ctcd-Kmeans-B.log 2>&1
# time python LT_qa_recommendation.py KmeansCommunityDetection -r SVRRecommender -o results/Kmeans-B > logs/ctqr-Kmeans-B.log 2>&1

# echo Kmeans Bipartite w/ attribute
# time python LT_community_detection.py KmeansCommunityDetection --attribute -o results/Kmeans-B-A > logs/ctcd-Kmeans-B-A.log 2>&1
# time python LT_qa_recommendation.py KmeansCommunityDetection -r SVRRecommender --attribute -o results/Kmeans-B-A > logs/ctqr-Kmeans-B-A.log 2>&1

# echo Kmeans Total w/o attribute
# time python CT_recommendation.py -r SVRRecommender -o results/Kmeans-T > logs/ctr-Kmeans-T.log 2>&1

# echo Kmeans Total w/ attribute
# time python CT_recommendation.py -r SVRRecommender --attribute -o results/Kmeans-T-A > logs/ctr-Kmeans-T-A.log 2>&1

# echo Each Item
# time python CT_recommendation.py -r SVRRecommender --EI -o results/EI > logs/ctr-EI.log 2>&1

# ------------- Clustered Tail -------------
# beta_list=(0.03125)
# method_list=(QUBOBipartiteProjectedCommunityDetection)
# cut_list=(0.01 0.02 0.025 0.03 0.04 0.05 0.100)
# for method in ${method_list[*]}
# do
#   for beta in ${beta_list[*]}
#   do
#     for cut in ${cut_list[*]}
#     do
#       tag=Cascade-$method-$beta-$cut
#       echo $tag
#       # time python LT_community_detection.py $method --attribute -b $beta -c $cut -o results/$tag > logs/ctcd-$tag.log 2>&1
#       time python LT_qa_recommendation.py $method -r LRRecommender --attribute -b $beta -c $cut -o results/$tag > logs/ctqr-$tag.log 2>&1
#     done
#   done
# done