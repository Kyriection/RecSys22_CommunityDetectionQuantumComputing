import numpy as np
from sklearn.linear_model import LinearRegression

from recsys.Recommenders.BaseRecommender import BaseRecommender
from utils.DataIO import DataIO

class AdaptiveClustering(BaseRecommender):
    RECOMMENDER_NAME = "AdaptiveClustering"

    def __init__(self, URM_train):
        super(AdaptiveClustering, self).__init__(URM_train)
        n_users, n_items = URM_train.shape
        self.n_users = n_users
        self.n_items = n_items
        self.scores = np.zeros((n_items, n_users))
        # self.models = [sklearn.linear_model.LinearRegression() for i in n_items]


    def fit(self, user_related_variables, groups: list = None):
      if groups is None: # EI
        groups = [[i] for i in range(self.n_items)]
      # prepare data
      X = [[] for i in range(self.n_items)]
      Y = [[] for i in range(self.n_items)]
      rows, cols = self.URM_train.nonzero()
      for row, col in zip(rows, cols):
         X[col].append(user_related_variables[row])
         Y[col].append(self.URM_train[row, col])
      # model fit & predict
      model = LinearRegression()
      for i in range(self.n_items):
        x = []
        y = []
        for j in groups[i]:
          x += X[j]
          y += Y[j]
        if len(x) == 0:
          print(f'[Warning] no data for item {i}.')
          self.scores[i] = np.ones(self.n_users) * 3.0
        else:
          # print(f'model for item {i} fit with group_size {len(groups[i])}, data_size {len(y)}')
          model.fit(x, y)
          self.scores[i] = model.predict(user_related_variables)
        # self.models[i].fit(x, y)
        # self.scores[i] = self.models[i].predict(users_feat)
      self.scores[self.scores < 1.0] = 1.0
      self.scores[self.scores > 5.0] = 5.0


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        if items_to_compute is None:
            return self.scores.T[user_id_array]
        else:
            return self.scores[items_to_compute].T[user_id_array]


    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"scores": self.scores}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")
