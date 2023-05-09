import logging
from tqdm import tqdm

import sklearn
import numpy as np
from typing import List

from CommunityDetection import Community
from recsys.Recommenders.BaseRecommender import BaseRecommender
from recsys.Recommenders.AdaptiveClustering import AdaptiveClustering
from utils.DataIO import DataIO
from utils.urm import get_community_urm
from utils.derived_variables import create_derived_variables

def KNN(item, item_related_variables, I_quantity, criterion:int = 0) -> List[int]:
    if criterion == 0:
        return [item]
    n_items = len(item_related_variables)
    distance = np.zeros(n_items)
    for i in range(n_items):
        distance[i] = np.sqrt(np.sum((item_related_variables[i] - item_related_variables[item])**2))
    item_rank_id = sorted(range(n_items), key=lambda x:distance[x])
    group = []
    group_size = 0
    # for i in tqdm(range(n_items)):
    for i in range(n_items):
        group.append(item_rank_id[i])
        group_size += I_quantity[item_rank_id[i]]
        if group_size >= criterion:
            break
    return group

def normalization(variables):
    min_val = np.min(variables, axis=0)
    max_val = np.max(variables, axis=0)
    _range = max_val - min_val
    _range[_range <= 0] = 1
    return (variables - min_val) / _range

def standardization(variables):
    mu = np.mean(variables, axis=0)
    sigma = np.std(variables, axis=0)
    sigma[sigma <= 0] = 1
    return (variables - mu) / sigma


def create_related_variables(urm, icm, ucm):
    C_aver_rating, C_quantity, C_seen_popularity, C_seen_rating,\
    I_aver_rating, I_quantity, I_likability = create_derived_variables(urm)
    item_related_variables = np.hstack([
        I_aver_rating.reshape((-1, 1)),
        I_quantity.reshape((-1, 1)),
        # I_likability.reshape((-1, 1)),
        icm.toarray(),
    ])
    user_related_variables = np.hstack([
        C_aver_rating.reshape((-1, 1)),
        C_quantity.reshape((-1, 1)),
        # C_seen_popularity.reshape((-1, 1)),
        # C_seen_rating.reshape((-1, 1)),
        ucm.toarray(),
    ])
    item_related_variables = normalization(item_related_variables)
    user_related_variables = normalization(user_related_variables)
    # item_related_variables = standardization(item_related_variables)
    # user_related_variables = standardization(user_related_variables)
    return item_related_variables, user_related_variables




class CommunityDetectionAdaptiveClustering(BaseRecommender):
    """
    EI: class(urm, ucm, icm, criterion=0, communities=None)
    AC: class(urm, ucm, icm, criterion, communities=None)
    CD_AC: class(urm, ucm, icm, criterion, communities)
    """
    RECOMMENDER_NAME = "CommunityDetectionAdaptiveClustering"

    def __init__(self, URM_train, ucm, icm, criterion: int = 0, communities: List[Community] = None):
        super(CommunityDetectionAdaptiveClustering, self).__init__(URM_train)
        self.ucm = ucm
        self.icm = icm
        self.criterion = criterion
        n_users, n_items = URM_train.shape
        self.n_users = n_users
        self.n_items = n_items
        self.scores = np.zeros((n_items, n_users))
        if communities is None:
            user_index = np.arange(self.n_users)
            item_index = np.arange(self.n_items)
            user_mask = np.zeros(self.n_users)
            item_mask = np.zeros(self.n_items)
            user_mask[user_index] = 1
            item_mask[item_index] = 1
            user_mask = user_mask.astype(bool)
            item_mask = item_mask.astype(bool)
            self.communities = [Community(user_index, item_index, user_mask, item_mask)]
        else:
            self.communities = communities
        # self.recommendors = []


    def fit(self):
        for community in self.communities:
            c_urm, _, _, c_icm, c_ucm = get_community_urm(self.URM_train, community=community, filter_users=False, remove=True, ucm=self.ucm, icm=self.icm)
            # c_urm_train_last_test = merge_sparse_matrices(c_urm_train, c_urm_validation)
            # print(c_urm.shape, c_icm.shape, c_ucm.shape)
            logging.info('create_related_variables.')
            I_quantity = np.ediff1d(c_urm.tocsc().indptr) # count of each colum
            item_related_variables, user_related_variables = create_related_variables(c_urm, c_icm, c_ucm)
            logging.info('AdaptiveClustering.')
            recommendor = AdaptiveClustering(c_urm)
            items = community.items
            # compute groups
            logging.info('compute groups.')
            groups = [KNN(item, item_related_variables, I_quantity, self.criterion) for item in items]
            # fit
            logging.info('fit')
            recommendor.fit(user_related_variables, item_related_variables, groups)
            # concate scores
            self.scores[items] = recommendor.scores


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        if items_to_compute is None:
            return self.scores.T[user_id_array]
        else:
            return self.scores[items_to_compute].T[user_id_array]


    def save_model(self, folder_path, file_name = None):
        # for recommendor in self.recommendors:
          # recommendor.save_model(folder_path, file_name)

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"scores": self.scores}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")
