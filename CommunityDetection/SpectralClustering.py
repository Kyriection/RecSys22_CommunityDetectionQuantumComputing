import time

import numpy as np
import sklearn.cluster
import scipy.sparse as sp
# from sklearn.cluster import SpectralClustering

from CommunityDetection.BaseCommunityDetection import BaseCommunityDetection
from utils.DataIO import DataIO


class SpectralClustering(BaseCommunityDetection):
    is_qubo = False
    filter_items = False
    name = 'SpectralClustering'

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(SpectralClustering, self).__init__(urm, *args, **kwargs)
        self.icm = icm
        self.ucm = ucm


    def save_model(self, folder_path, file_name):
        data_dict_to_save = {
            '_fit_time': self._fit_time,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

    def run(self) -> [np.ndarray, np.ndarray, float]:
        start_time = time.time()

        clustering = sklearn.cluster.SpectralClustering(
            n_clusters=2,
            eigen_solver='arpack',
            # eigen_solver='lobpcg',
            # eigen_solver='amg',
            random_state=0,
            assign_labels='discretize',
            # affinity = 'precomputed', 
            # n_init=1000,
        )
        n_users, n_items = self.urm.shape

        X = sp.hstack((self.urm, self.ucm))
        # urm: np.ndarray = self.urm.toarray()
        # urm[np.isnan(urm)] = 0
        # urm.replace([-np.inf, np.inf], 0)
        # urm = (urm - np.mean(urm)) / np.std(urm)
        try:
            users = clustering.fit_predict(X)
        except Exception as e:
            users = np.ones(n_users)
            print('[Error] spectral clustering: ', e)

        run_time = time.time() - start_time

        assert len(users) == n_users, "Output of SpectralClustering doesn't fit users"
        return users, np.zeros(n_items), run_time
    
    def fit(self, *args, **kwargs):
        start_time = time.time()

        # nothing

        self._fit_time = time.time() - start_time