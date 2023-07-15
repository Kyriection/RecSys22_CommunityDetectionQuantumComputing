import time

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

from CommunityDetection.BaseCommunityDetection import BaseCommunityDetection
from utils.DataIO import DataIO


class KmeansBipartiteCommunityDetection(BaseCommunityDetection):
    is_qubo = False
    filter_items = False
    name = 'KmeansBipartiteCommunityDetection'
    attribute: bool = False

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(KmeansBipartiteCommunityDetection, self).__init__(urm, *args, **kwargs)
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
            if KmeansBipartiteCommunityDetection.attribute:
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
        KmeansBipartiteCommunityDetection.attribute = attribute
