'''
Author: Kaizyn
Date: 2023-01-11 13:45:23
LastEditTime: 2023-01-13 21:18:12
'''
# import logging
import time

import numpy as np

from CommunityDetection.Communities import Communities
from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection

# logging.basicConfig(level=logging.INFO)

class QUBOLongTailCommunityDetection(QUBOCommunityDetection):
    filter_items = False
    name = 'QUBOLongTailCommunityDetection'
    # n_all_users = -1

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(QUBOLongTailCommunityDetection, self).__init__(urm, *args, **kwargs)
        # self.set_n_all_users(urm.shape[0])
        self.icm = icm
        self.ucm = ucm
        self.W = None

    def fit(self, weighted=True, threshold=None):
        start_time = time.time()

        W = self.urm * self.urm.T

        if not weighted:
            W = (W > 0) * 1

        W.setdiag(0)
        W.eliminate_zeros()

        k = W.sum(axis=1)
        m = k.sum() // 2

        P = k @ k.T / (2 * m)

        B = W - P
        C_quantity = np.ediff1d(self.urm.tocsr().indptr)
        C_quantity = C_quantity / np.max(C_quantity) # normalization
        n_users, n_items = self.urm.shape
        # ratio = n_users / self.n_all_users
        T = 12
        diag = np.exp(C_quantity * T)
        # diag = (diag - diag.mean()) * ratio**2
        diag = (diag - diag.mean())
        # logging.info(f'B_max={np.max(B)}, diag_max={np.max(diag)}')
        for i in range(n_users):
            B[i, i] += diag[i]

        if threshold is not None:
            B[np.abs(B) < threshold] = 0

        self.W = W
        self._Q = -B / m  # Normalized QUBO matrix

        self._fit_time = time.time() - start_time

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        users, _ = super(QUBOLongTailCommunityDetection,
                         QUBOLongTailCommunityDetection).get_comm_from_sample(sample, n_users)
        return users, np.zeros(n_items)
    
    '''
    @staticmethod
    def set_n_all_users(n_users: int):
        if QUBOLongTailCommunityDetection.n_all_users == -1:
            QUBOLongTailCommunityDetection.n_all_users = n_users
            print(f"{QUBOLongTailCommunityDetection.name}: set n_all_users={n_users}")
    '''
    def get_graph_cut(self, communities: Communities):
        cut = 0.0
        rows, cols = self.W.nonzero()
        for row, col in zip(rows, cols):
            if communities.user_mask[row] != communities.user_mask[col]:
                cut += self.W[row, col]
        all = self.W.sum()
        # print(f'cut/all={cut}/{all}={cut/all}')
        return cut, all