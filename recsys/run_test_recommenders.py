#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2018

@author: Anonymous
"""

import traceback, os, shutil

from recsys.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender

from recsys.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample
from recsys.Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from recsys.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from recsys.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping


def write_log_string(log_file, string):
    log_file.write(string)
    log_file.flush()


def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):

    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object


def run_recommender(recommender_class):
    temp_save_file_folder = "./result_experiments/__temp_model/"

    if not os.path.isdir(temp_save_file_folder):
        os.makedirs(temp_save_file_folder)

    try:
        dataset_object = Movielens1MReader()

        dataSplitter = DataSplitter_leave_k_out(dataset_object, k_out_value=2)

        dataSplitter.load_data()
        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
        ICM_all = dataSplitter.get_loaded_ICM_dict()["ICM_genres"]
        UCM_all = dataSplitter.get_loaded_UCM_dict()["UCM_all"]

        write_log_string(log_file, "On Recommender {}\n".format(recommender_class))

        recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)

        if isinstance(recommender_object, Incremental_Training_Early_Stopping):
            fit_params = {"epochs": 15}
        else:
            fit_params = {}

        recommender_object.fit(**fit_params)

        write_log_string(log_file, "Fit OK, ")



        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen = True)
        results_df, results_run_string = evaluator.evaluateRecommender(recommender_object)

        write_log_string(log_file, "EvaluatorHoldout OK, ")



        evaluator = EvaluatorNegativeItemSample(URM_test, URM_train, [5], exclude_seen = True)
        _, _ = evaluator.evaluateRecommender(recommender_object)

        write_log_string(log_file, "EvaluatorNegativeItemSample OK, ")



        recommender_object.save_model(temp_save_file_folder, file_name="temp_model")

        write_log_string(log_file, "save_model OK, ")



        recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)
        recommender_object.load_model(temp_save_file_folder, file_name="temp_model")

        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen = True)
        result_df_load, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

        assert results_df.equals(result_df_load), "The results of the original model should be equal to that of the loaded one"

        write_log_string(log_file, "load_model OK, ")



        shutil.rmtree(temp_save_file_folder, ignore_errors = True)

        write_log_string(log_file, " PASS\n")
        write_log_string(log_file, results_run_string + "\n\n")



    except Exception as e:

        print("On Recommender {} Exception {}".format(recommender_class, str(e)))
        log_file.write("On Recommender {} Exception {}\n\n\n".format(recommender_class, str(e)))
        log_file.flush()

        traceback.print_exc()


from recsys.Recommenders.Recommender_import_list import *


if __name__ == '__main__':


    log_file_name = "./result_experiments/run_test_recommender.txt"


    recommender_list = [
        # Random,
        # TopPop,
        # GlobalEffects,
        # UserKNNCFRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCBFRecommender,
        # ItemKNNCBFRecommender,
        # ItemKNN_CFCBF_Hybrid_Recommender,
        # UserKNN_CFCBF_Hybrid_Recommender,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # MatrixFactorization_AsySVD_Cython,
        # PureSVDRecommender,
        # NMFRecommender,
        # IALSRecommender,
        # EASE_R_Recommender,
        # LightFMCFRecommender,
        # LightFMUserHybridRecommender,
        # LightFMItemHybridRecommender,
        MultVAERecommender,
    ]

    log_file = open(log_file_name, "w")



    for recommender_class in recommender_list:
        run_recommender(recommender_class)
    #
    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(run_dataset, dataset_list)

