import numpy as np

from CommunityDetection.Communities import Communities
from CommunityDetection.CommunityDetectionRecommender import CommunityDetectionRecommender
from recsys.Recommenders.BaseRecommender import BaseRecommender
from utils.types import List, NDArray

def calc_num_iters(communities_list: List[Communities]):
    n_iter = communities_list[0].num_iters
    for communites in communities_list:
        n_iter = min(n_iter, communites.num_iters)
    return n_iter

class HybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "HybridRecommender"

    def __init__(self, URM_train, communities_list: List[Communities], recommenders_list: List[List[BaseRecommender]], n_iter=None,
                 verbose=True, communities_weight: List[float] = None):
        super(HybridRecommender, self).__init__(URM_train, verbose=verbose)
        
        if n_iter is None:
            n_iter = calc_num_iters(communities_list)
        self.n_iter: int = n_iter
        self.community_detection_recommenders = [
            CommunityDetectionRecommender(URM_train, communities=communities, recommenders=recommenders, n_iter=n_iter)
            for communities, recommenders in zip(communities_list, recommenders_list)
        ]
        self.communities_weight = communities_weight or [1.0 / len(communities_list)] * len(communities_list) # average

    def _compute_item_score(self, user_id_array: NDArray, items_to_compute=None):
        item_scores = np.zeros((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)

        for idx, recommender in enumerate(self.community_detection_recommenders):
            item_scores += self.communities_weight[idx] * recommender._compute_item_score(user_id_array, items_to_compute)

        return item_scores