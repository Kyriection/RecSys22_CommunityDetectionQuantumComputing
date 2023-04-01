import time

import numpy as np
import sklearn.cluster
# from sklearn.cluster import SpectralClustering

from CommunityDetection.BaseCommunityDetection import BaseCommunityDetection
from utils.DataIO import DataIO


class SpectralClustering(BaseCommunityDetection):
    is_qubo = False
    filter_items = False
    name = 'HierarchicalClustering'

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(SpectralClustering, self).__init__(urm, *args, **kwargs)


    def save_model(self, folder_path, file_name):
        data_dict_to_save = {
            '_fit_time': self._fit_time,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

    def run(self) -> [np.ndarray, np.ndarray, float]:
        start_time = time.time()

        clustering = sklearn.cluster.SpectralClustering(
            n_clusters=2, assign_labels='discretize', random_state=1)
        n_users, n_items = self.urm.shape

        if n_users < 2:
            users = np.ones(n_users)
        else:
            users = clustering.fit_predict(self.urm.toarray())

        run_time = time.time() - start_time

        assert len(users) == n_users, "Output of SpectralClustering doesn't fit users"
        return users, np.zeros(n_items), run_time
    
    def fit(self, *args, **kwargs):
        start_time = time.time()

        # nothing

        self._fit_time = time.time() - start_time