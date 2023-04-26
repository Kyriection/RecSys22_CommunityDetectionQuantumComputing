import numpy as np
import pandas as pd

from recsys.Recommenders.BaseRecommender import BaseRecommender
from recsys.Recommenders.AdaptiveClustering import AdaptiveClustering

class AC_Evaluator(object):
    EVALUATOR_NAME = "AC_Evaluator"

    def __init__(self, URM_test):
        super(AC_Evaluator, self).__init__()
        self.URM_test = URM_test
        n_users, n_items = URM_test.shape
        self.n_users = n_users
        self.n_items = n_items


    def evaluateRecommender(self, recommender_object: BaseRecommender):
        user_id_array = np.arange(self.n_users)
        predict_scores = recommender_object._compute_item_score(user_id_array)

        MAE = np.zeros(self.n_items)
        RMSE = np.zeros(self.n_items)
        num_rating = np.ediff1d(self.URM_test.tocsc().indptr) # count of each colum
        # print('num_rating:', num_rating)
        # print('num_rating <= 1: ', (num_rating <= 1).sum()) # 758

        rows, cols = self.URM_test.nonzero()
        for row, col in zip(rows, cols):
            diff = abs(self.URM_test[row, col] - predict_scores[row, col])
            MAE[col] += diff
            RMSE[col] += diff ** 2

        num_rating_all = np.sum(num_rating)
        MAE_all = np.sum(MAE) / num_rating_all
        RMSE_all = np.sqrt(np.sum(RMSE) / num_rating_all)
        num_rating[num_rating == 0] = 1
        MAE /= num_rating
        RMSE = np.sqrt(RMSE / num_rating)

        '''
        results_df = pd.DataFrame(columns=results_dict[self.cutoff_list[0]].keys(),
                                  index=self.cutoff_list)
        results_df.index.rename("cutoff", inplace = True)

        for cutoff in results_dict.keys():
            results_df.loc[cutoff] = results_dict[cutoff]

        try:
            results_run_string = get_result_string_df(results_df)
        except TypeError:
            if self._n_users_evaluated > 0:
                raise TypeError
            results_run_string = ""

        return results_df, results_run_string
        '''

        return MAE, RMSE, MAE_all, RMSE_all

