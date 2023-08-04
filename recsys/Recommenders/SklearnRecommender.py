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

from recsys.Recommenders.BaseRecommender import BaseRecommender
from utils.DataIO import DataIO

logging.basicConfig(level=logging.INFO)

def get_recommender_class(sklearn_model, recommender_name: str):
    class SklearnRecommender(BaseRecommender):

        RECOMMENDER_NAME = recommender_name
        LIMIT_MIN = 0.0
        LIMIT_MAX = 5.0

        def __init__(self, URM_train, ucm, icm, user_id_array: Optional[list] = None):
            super(SklearnRecommender, self).__init__(URM_train)
            self.ucm = ucm
            self.icm = icm
            self.model: sklearn_model = None
            if user_id_array is None:
                self.user_id_array = list(range(self.n_users))
            else:
                self.user_id_array = user_id_array
            logging.debug(f'len(self.user_id_array)={len(self.user_id_array)}')
            self.scores = None


        def fit(self):
            start_time = time.time()

            rows, cols = self.URM_train.nonzero()
            x = [sp.hstack((self.ucm[row], self.icm[col])).A.flatten() for row, col in zip(rows, cols)]
            y = [self.URM_train[row, col] for row, col in zip(rows, cols)]
            if len(y) > 0:
                self.model = sklearn_model()
                self.model.fit(x, y)

            logging.info(f'fit {len(self.user_id_array)} users with {len(y)} ratings, cost time {time.time() - start_time}s.')


        def predict(self, user_id: int, items_id = None):
            user = list(self.user_id_array).index(user_id)
            # user = np.where(self.user_id_array == user_id)[0][0]
            if items_id is None:
                items_id = range(self.n_items)
            elif isinstance(items_id, int):
                items_id = [items_id]
            if self.scores is not None:
                return self.scores[user, items_id].copy()
            if self.model is None:
                return (self.LIMIT_MIN + self.LIMIT_MAX) / 2 * np.ones(len(items_id), dtype=np.float32)

            x = [sp.hstack((self.ucm[user_id], self.icm[i])).A.flatten() for i in items_id]
            scores = self.model.predict(x)
            scores[scores < self.LIMIT_MIN] = self.LIMIT_MIN
            scores[scores > self.LIMIT_MAX] = self.LIMIT_MAX
            return scores


        def predict_all(self):
            if self.scores is not None:
                return
            n_users = len(self.user_id_array)
            self.scores = (self.LIMIT_MIN + self.LIMIT_MAX) / 2 * np.ones((n_users, self.n_items), dtype=np.float32)
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

        @staticmethod
        def set_limit(limit_min, limit_max):
            logging.info(f'Recommender set limit [{limit_min}, {limit_max}].')
            SklearnRecommender.LIMIT_MIN = limit_min
            SklearnRecommender.LIMIT_MAX = limit_max
    
    return SklearnRecommender
