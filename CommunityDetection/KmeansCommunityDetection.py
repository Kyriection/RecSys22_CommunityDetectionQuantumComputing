import time

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

from CommunityDetection.BaseCommunityDetection import BaseCommunityDetection
from utils.DataIO import DataIO


class KmeansCommunityDetection(BaseCommunityDetection):
    is_qubo = False
    filter_items = False
    name = 'KmeansCommunityDetection'
    attribute: bool = False

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(KmeansCommunityDetection, self).__init__(urm, *args, **kwargs)
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

        kmeans = KMeans(n_clusters=2, random_state=0)
        n_users, n_items = self.urm.shape
        n_genres = self.icm.shape[1]

        if n_users < 2:
            users = np.ones(n_users)
        else:
            X = self.urm
            if KmeansCommunityDetection.attribute:
                X = sp.hstack((self.urm, self.ucm))
            users = kmeans.fit_predict(X)

        run_time = time.time() - start_time

        assert len(users) == n_users, "Output of KMeans doesn't fit users"
        return users, np.zeros(n_items), run_time
    
    def fit(self, *args, **kwargs):
        start_time = time.time()

        # nothing

        self._fit_time = time.time() - start_time

    @staticmethod
    def set_attribute(attribute: bool):
        KmeansCommunityDetection.attribute = attribute

    """
    def get_Q_adjacency(self):
        BQM = dimod.BinaryQuadraticModel(self._Q, vartype=dimod.BINARY)
        return dimod.to_networkx_graph(BQM)

    @staticmethod
    def get_comm_from_sample(sample, n, **kwargs):
        n_features = len(sample)
        comm = np.zeros(n_features, dtype=int)
        for k, v in sample.items():
            if v == 1:
                ind = int(k)
                comm[ind] = 1

        return comm[:n], comm[n:]
    """
