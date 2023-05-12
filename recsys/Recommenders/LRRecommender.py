#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Anonymous
"""
import logging
import time
import tqdm
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression

from recsys.Recommenders.BaseRecommender import BaseRecommender
from utils.DataIO import DataIO

logging.basicConfig(level=logging.INFO)

class LRRecommender(BaseRecommender):
    """Linear Regression recommender"""

    RECOMMENDER_NAME = "LRRecommender"

    def __init__(self, URM_train, ucm, icm, user_id_array: Optional[list] = None):
        super(LRRecommender, self).__init__(URM_train)
        self.ucm = ucm
        self.icm = icm
        self.model: LinearRegression = None
        if user_id_array is None:
            self.user_id_array = list(range(self.n_users))
        else:
            self.user_id_array = user_id_array
        logging.debug(f'len(self.user_id_array)={len(self.user_id_array)}')
        self.scores = None


    def fit(self):
        start_time = time.time()

        rows, cols = self.URM_train.nonzero()
        # x = [np.hstack((self.ucm[row], self.icm[col])) for row, col in zip(rows, cols)]
        x = [sp.hstack((self.ucm[row], self.icm[col])).A.flatten() for row, col in zip(rows, cols)]
        y = [self.URM_train[row, col] for row, col in zip(rows, cols)]
        
        if len(y) == 0:
            pass
        else:
            self.model = LinearRegression()
            self.model.fit(x, y)
        logging.info(f'fit {len(self.user_id_array)} users with {len(y)} ratings, cost time {time.time() - start_time}s.')


    def predict(self, user_id: int, items_id):
        user = self.user_id_array.find(user_id)
        assert user != -1, f'[Error] {user_id} is not in recommender.user_id_array.'
        if items_id is None:
            items_id = range(self.n_items)
        elif isinstance(int, items_id):
            items_id = [items_id]
        if self.scores is not None:
            return self.scores[user_id][items_id].copy()

        x = [sp.hstack((self.ucm[user], self.icm[i])).A.flatten() for i in items_id]
        scores = self.model.predict(x)
        scores[scores < 1.0] = 1.0
        scores[scores > 5.0] = 5.0
        return scores


    def predict_all(self):
        if self.scores is not None:
            return
        n_users = len(self.user_id_array)
        self.scores = np.ones((n_users, self.n_items), dtype=np.float32) * 3
        if self.model is not None:
            for i, user in tqdm.tqdm(enumerate(self.user_id_array)):
                self.scores[i] = self.predict(user)
        return self.scores

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        self.predict_all()
        fit_mask = np.in1d(user_id_array, self.user_id_array, assume_unique=True)
        choose_mask = np.in1d(self.user_id_array, user_id_array, assume_unique=True)
        item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
        logging.debug(f'{len(user_id_array)}, {len(self.user_id_array)}, {fit_mask.shape}, {choose_mask.shape}, {item_scores.shape}, {self.scores.shape}')
        item_scores[fit_mask, :] = self.scores[choose_mask, :].copy()
        if items_to_compute is not None:
            item_scores = item_scores[:, items_to_compute]
        return item_scores


    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"scores": self.scores}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")
