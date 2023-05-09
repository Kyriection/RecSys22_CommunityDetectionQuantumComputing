#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Anonymous
"""
import logging

import numpy as np
from sklearn.linear_model import LinearRegression

from recsys.Recommenders.BaseRecommender import BaseRecommender
from utils.DataIO import DataIO


class LRRecommender(BaseRecommender):
    """Linear Regression recommender"""

    RECOMMENDER_NAME = "LRRecommender"

    def __init__(self, URM_train, user, ucm, icm):
        super(LRRecommender, self).__init__(URM_train)
        self.user = user
        self.ucm = ucm
        self.icm = icm
        self.model = LinearRegression()
        self.scores = None


    def fit(self):
        rows, cols = self.URM_train.nonzero()
        x = [np.hstack((self.ucm[row], self.icm[col])) for row, col in zip(rows, cols)]
        y = [self.URM_train[row, col] for row, col in zip(rows, cols)]
        self.model.fit(x, y)
        x = [np.hstack((self.ucm[self.user], self.icm[i])) for i in range(self.n_items)]
        self.scores = self.model.predict(x)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        if items_to_compute is not None:
            n_users = len(user_id_array)
            item_scores = - np.ones((n_users, self.n_items), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = self.scores[user_id_array, items_to_compute].copy()
        else:
            return self.scores[user_id_array].copy()


    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"scores": self.scores}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")
