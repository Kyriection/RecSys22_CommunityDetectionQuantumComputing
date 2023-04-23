'''
Author: Kaizyn
Date: 2023-01-11 13:45:23
LastEditTime: 2023-01-13 21:18:12
'''
import time

import numpy as np

from CommunityDetection.Communities import Communities
from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class QUBOBipartiteProjectedItemCommunityDetection(QUBOCommunityDetection):
    filter_users = False
    name = 'QUBOBipartiteProjectedItemCommunityDetection'

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(QUBOBipartiteProjectedItemCommunityDetection, self).__init__(urm, *args, **kwargs)
        self.icm = icm
        self.ucm = ucm
        self.W = None

    def fit(self, weighted=True, threshold=None):
        start_time = time.time()

        W = self.urm.T * self.urm

        if not weighted:
            W = (W > 0) * 1

        W.setdiag(0)
        W.eliminate_zeros()

        k = W.sum(axis=1)
        m = k.sum() // 2

        P = k @ k.T / (2 * m)

        B = W - P

        if threshold is not None:
            B[np.abs(B) < threshold] = 0

        self.W = W
        self._Q = -B / m  # Normalized QUBO matrix

        self._fit_time = time.time() - start_time

    @staticmethod
    def get_comm_from_sample(sample, n_users=0, n_items=0):
        items, _ = super(QUBOBipartiteProjectedItemCommunityDetection,
                         QUBOBipartiteProjectedItemCommunityDetection).get_comm_from_sample(sample, n_items)
        return np.zeros(n_users), items