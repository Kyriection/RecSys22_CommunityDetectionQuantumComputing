#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import time

import numpy as np
import scipy.sparse as sps
import time, sys, copy
import pandas as pd

from recsys.Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

def _create_empty_metrics_dict(n_users):
    empty_dict = {}
    for i in range(n_users):
        empty_dict[i] = {
            'MAE': 0.0,
            'MSE': 0.0,
            'num_rating': 0,
        }
    return empty_dict

def _remove_item_interactions(URM, item_list):

    URM = sps.csc_matrix(URM.copy())

    for item_index in item_list:

        start_pos = URM.indptr[item_index]
        end_pos = URM.indptr[item_index+1]

        URM.data[start_pos:end_pos] = np.zeros_like(URM.data[start_pos:end_pos])

    URM.eliminate_zeros()
    URM = sps.csr_matrix(URM)

    return URM


class EvaluatorSeparate(object):
    """Abstract EvaluatorSeparate"""

    EVALUATOR_NAME = "Evaluator_Base_Class"

    def __init__(self, URM_test_list, min_ratings_per_user=1, exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None,
                 verbose=True):
        """
        ignore_users/items: ignore users/items not in community
        users_to_evaluate: users with more than 1 rating in URM
        """

        super(EvaluatorSeparate, self).__init__()

        self.verbose = verbose

        if ignore_items is None:
            self.ignore_items_flag = False
            self.ignore_items_ID = np.array([])
        else:
            self._print("Ignoring {} Items".format(len(ignore_items)))
            self.ignore_items_flag = True
            self.ignore_items_ID = np.array(ignore_items)

        self.max_cutoff = 1 # no need for cutoff

        self.min_ratings_per_user = min_ratings_per_user
        self.exclude_seen = exclude_seen

        if not isinstance(URM_test_list, list):
            self.URM_test = URM_test_list.copy()
            URM_test_list = [URM_test_list]
        else:
            raise ValueError("List of URM_test not supported")

        self.diversity_object = diversity_object

        self.n_users, self.n_items = URM_test_list[0].shape

        # Prune users with an insufficient number of ratings
        # During testing CSR is faster
        self.URM_test_list = []
        users_to_evaluate_mask = np.zeros(self.n_users, dtype=np.bool)

        for URM_test in URM_test_list:

            URM_test = _remove_item_interactions(URM_test, self.ignore_items_ID)

            URM_test = sps.csr_matrix(URM_test)
            self.URM_test_list.append(URM_test)

            rows = URM_test.indptr
            numRatings = np.ediff1d(rows)
            new_mask = numRatings >= min_ratings_per_user

            users_to_evaluate_mask = np.logical_or(users_to_evaluate_mask, new_mask)

        if not np.all(users_to_evaluate_mask):
            self._print("Ignoring {} ({:4.1f}%) Users that have less than {} test interactions".format(np.sum(users_to_evaluate_mask),
                                                                                                     100*np.sum(np.logical_not(users_to_evaluate_mask))/len(users_to_evaluate_mask), min_ratings_per_user))

        self.users_to_evaluate = np.arange(self.n_users)[users_to_evaluate_mask]

        if ignore_users is not None:
            self._print("Ignoring {} Users".format(len(ignore_users)))
            self.ignore_users_ID = np.array(ignore_users)
            self.users_to_evaluate = set(self.users_to_evaluate) - set(ignore_users)
        else:
            self.ignore_users_ID = np.array([])

        self.users_to_evaluate = list(self.users_to_evaluate)

        # Those will be set at each new evaluation
        self._start_time = np.nan
        self._start_time_print = np.nan
        self._n_users_evaluated = np.nan

        # self._print(f"len(users_to_evaluate)={len(self.users_to_evaluate)}")


    def _print(self, string):

        if self.verbose:
            print("{}: {}".format(self.EVALUATOR_NAME, string))


    def evaluateRecommender(self, recommender_object):
        """
        :param recommender_object: the trained recommender object, a BaseRecommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :return results_df: dataframe with index the cutoff and columns the metric
        :return results_run_string: printable result string
        """
        start_time = time.time()

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        self._start_time = time.time()
        self._start_time_print = time.time()
        self._n_users_evaluated = 0

        # results_dict = self._run_evaluation_on_selected_users(recommender_object, self.users_to_evaluate)
        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        results_dict = _create_empty_metrics_dict(self.n_users)
        results_dict = self._compute_metrics_on_recommendation_list(recommender_object, results_dict)

        if self._n_users_evaluated > 0:
            pass
        else:
            self._print(f"self.n_users={self.n_users}, self._n_users_evaluated={self._n_users_evaluated}")
            self._print("WARNING: No users had a sufficient number of relevant items")

        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()

        results_df = pd.DataFrame(columns=results_dict[0].keys(),
                                  index=list(range(self.n_users)))
        results_df.index.rename("user", inplace = True)

        for user in results_dict.keys():
            results_df.loc[user] = results_dict[user]

        logging.info(f'evaluateRecommender cost time: {time.time() - start_time}')
        return results_df, ""



    def get_user_relevant_items(self, user_id):
        """
        return array[items id] which has value in URM_test
        """

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in getting relevant items"

        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]


    def get_user_test_ratings(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in relevant items ratings"

        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]



    def _compute_metrics_on_recommendation_list(self, recommender_object, results_dict):
        
        for test_user in self.users_to_evaluate:

            relevant_items = self.get_user_relevant_items(test_user)

            # Add the RMSE to the global object, no need to loop through the various cutoffs
            relevant_items_rating = self.get_user_test_ratings(test_user)
            all_items_predicted_ratings = recommender_object.predict(test_user, relevant_items)
            assert len(all_items_predicted_ratings) == len(relevant_items), f'scores size not match: {len(all_items_predicted_ratings)} != {len(relevant_items)}'
            # global_RMSE_object = results_dict[self.cutoff_list[0]][EvaluatorMetrics.RMSE.value]
            # global_RMSE_object.add_recommendations(all_items_predicted_ratings, relevant_items, relevant_items_rating)
            diff = relevant_items_rating - all_items_predicted_ratings
            results_dict[test_user]['MAE'] = np.mean(np.abs(diff))
            results_dict[test_user]['MSE'] = np.mean(diff**2)
            results_dict[test_user]['num_rating'] = len(relevant_items)
            self._n_users_evaluated += 1



        if time.time() - self._start_time_print > 300 or self._n_users_evaluated==len(self.users_to_evaluate):

            elapsed_time = time.time()-self._start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            self._print("Processed {} ({:4.1f}%) in {:.2f} {}. Users per second: {:.0f}".format(
                          self._n_users_evaluated,
                          100.0* float(self._n_users_evaluated)/len(self.users_to_evaluate),
                          new_time_value, new_time_unit,
                          float(self._n_users_evaluated)/elapsed_time if elapsed_time > 0 else np.nan))

            sys.stdout.flush()
            sys.stderr.flush()

            self._start_time_print = time.time()


        return results_dict
