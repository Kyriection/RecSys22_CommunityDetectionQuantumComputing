import time

import numpy as np

from CommunityDetection.BaseCommunityDetection import BaseCommunityDetection
from utils.DataIO import DataIO


class QuantityDivision(BaseCommunityDetection):
    is_qubo = False
    filter_items = False
    name = 'QuantityDivision'
    T: int = 5

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(QuantityDivision, self).__init__(urm, *args, **kwargs)
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

        n_users, n_items = self.urm.shape
        C_quantity = np.ediff1d(self.urm.tocsr().indptr)
        C_quantity = C_quantity / np.max(C_quantity) # normalization
        diag = np.exp(C_quantity * self.T)
        diag /= np.sum(diag)
        diag -= diag.mean()
        users = diag > 0

        run_time = time.time() - start_time

        return users, np.zeros(n_items), run_time
    
    def fit(self, *args, **kwargs):
        start_time = time.time()
        # nothing
        self._fit_time = time.time() - start_time
    
    @staticmethod
    def set_T(T: int):
        QuantityDivision.T = T
