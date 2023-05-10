#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        self._start_time = time.time()
        self._start_time_print = time.time()
        self._n_users_evaluated = 0

        results_dict = self._run_evaluation_on_selected_users(recommender_object, self.users_to_evaluate)


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



    def _compute_metrics_on_recommendation_list(self, test_user_batch_array, recommended_items_batch_list, scores_batch, results_dict):

        assert len(recommended_items_batch_list) == len(test_user_batch_array), "{}: recommended_items_batch_list contained recommendations for {} users, expected was {}".format(
            self.EVALUATOR_NAME, len(recommended_items_batch_list), len(test_user_batch_array))

        assert scores_batch.shape[0] == len(test_user_batch_array), "{}: scores_batch contained scores for {} users, expected was {}".format(
            self.EVALUATOR_NAME, scores_batch.shape[0], len(test_user_batch_array))

        assert scores_batch.shape[1] == self.n_items, "{}: scores_batch contained scores for {} items, expected was {}".format(
            self.EVALUATOR_NAME, scores_batch.shape[1], self.n_items)


        # Compute recommendation quality for each user in batch
        for batch_user_index in range(len(scores_batch)):

            test_user = test_user_batch_array[batch_user_index]

            relevant_items = self.get_user_relevant_items(test_user)

            # Add the RMSE to the global object, no need to loop through the various cutoffs
            relevant_items_rating = self.get_user_test_ratings(test_user)
            all_items_predicted_ratings = scores_batch[batch_user_index]
            # global_RMSE_object = results_dict[self.cutoff_list[0]][EvaluatorMetrics.RMSE.value]
            # global_RMSE_object.add_recommendations(all_items_predicted_ratings, relevant_items, relevant_items_rating)
            MAE = 0.0
            MSE = 0.0
            num_rating = 0
            for i, item in enumerate(relevant_items):
                diff = relevant_items_rating[i] - all_items_predicted_ratings[item]
                MAE += np.abs(diff)
                MSE += diff**2
                num_rating += 1
            MAE /= num_rating
            MSE /= num_rating

            self._n_users_evaluated += 1
            results_dict[batch_user_index]['MAE'] = MAE
            results_dict[batch_user_index]['MSE'] = MSE
            results_dict[batch_user_index]['num_rating'] = num_rating



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







class EvaluatorSeparateHoldout(EvaluatorSeparate):
    """EvaluatorSeparateHoldout"""

    EVALUATOR_NAME = "EvaluatorSeparateHoldout"

    def __init__(self, URM_test_list, min_ratings_per_user=1, exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None,
                 verbose=True):


        super(EvaluatorSeparateHoldout, self).__init__(URM_test_list, diversity_object = diversity_object,
                                               min_ratings_per_user =min_ratings_per_user, exclude_seen=exclude_seen,
                                               ignore_items = ignore_items, ignore_users = ignore_users,
                                               verbose = verbose)





    def _run_evaluation_on_selected_users(self, recommender_object, users_to_evaluate, block_size = None):
        """
        every bach run block_size of users in users_to_evaluate(list)
        """

        if block_size is None:
            # Reduce block size if estimated memory requirement exceeds 4 GB
            block_size = min([1000, int(4*1e9*8/64/self.n_items), len(users_to_evaluate)])

        results_dict = _create_empty_metrics_dict(self.n_users)

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        # Start from -block_size to ensure it to be 0 at the first block
        user_batch_start = 0
        user_batch_end = 0

        while user_batch_start < len(users_to_evaluate):

            user_batch_end = user_batch_start + block_size
            user_batch_end = min(user_batch_end, len(users_to_evaluate))

            test_user_batch_array = np.array(users_to_evaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            recommended_items_batch_list, scores_batch = recommender_object.recommend(test_user_batch_array,
                                                                      remove_seen_flag=self.exclude_seen,
                                                                      cutoff = self.max_cutoff,
                                                                      remove_top_pop_flag=False,
                                                                      remove_custom_items_flag=self.ignore_items_flag,
                                                                      return_scores = True
                                                                     )

            results_dict = self._compute_metrics_on_recommendation_list(test_user_batch_array = test_user_batch_array,
                                                         recommended_items_batch_list = recommended_items_batch_list,
                                                         scores_batch = scores_batch,
                                                         results_dict = results_dict)


        return results_dict
