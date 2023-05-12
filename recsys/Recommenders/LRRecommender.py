#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Anonymous
"""
import logging
import time
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from recsys.Recommenders.BaseRecommender import BaseRecommender
from utils.DataIO import DataIO

logging.basicConfig(level=logging.INFO)

class LRRecommender(BaseRecommender):
    """Linear Regression recommender"""

    RECOMMENDER_NAME = "LRRecommender"

    def __init__(self, URM_train, ucm, icm, user_id_array: Optional[list] = None):
        super(LRRecommender, self).__init__(URM_train)
        self.ucm = ucm.toarray()
        self.icm = icm.toarray()
        self.model = LinearRegression()
        if user_id_array is None:
            self.user_id_array = list(range(self.n_users))
        else:
            self.user_id_array = user_id_array
        logging.debug(f'len(self.user_id_array)={len(self.user_id_array)}')
        self.scores = None


    def fit(self):
        start_time = time.time()

        rows, cols = self.URM_train.nonzero()
        x = [np.hstack((self.ucm[row], self.icm[col])) for row, col in zip(rows, cols)]
        y = [self.URM_train[row, col] for row, col in zip(rows, cols)]
        
        n_users = len(self.user_id_array)
        self.scores = np.ones((n_users, self.n_items), dtype=np.float32) * 3
        if len(y) == 0:
            pass
        else:
            self.model.fit(x, y)
            for i, user in enumerate(self.user_id_array):
                x = [np.hstack((self.ucm[user], self.icm[i])) for i in range(self.n_items)]
                self.scores[i] = self.model.predict(x)
            self.scores[self.scores < 1.0] = 1.0
            self.scores[self.scores > 5.0] = 5.0
        logging.info(f'fit {n_users} users with {len(y)} ratings, scores.shape{self.scores.shape}, cost time {time.time() - start_time}s.')


    def _compute_item_score(self, user_id_array, items_to_compute = None):

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
