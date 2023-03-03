'''
Author: Kaizyn
Date: 2023-01-11 13:45:23
LastEditTime: 2023-01-13 21:18:12
'''
import time

import numpy as np

from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection

def normalization(m):
    min_val = np.min(m)
    max_val = np.max(m)
    # assert min_val == 0.0
    return m / max_val

def get_similarity(m):
    W = m * m.T
    return normalization(W)


class QUBOProjectedCommunityDetection(QUBOCommunityDetection):
    filter_items = False
    name = 'QUBOProjectedCommunityDetection'
    alpha = 0.5

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(QUBOProjectedCommunityDetection, self).__init__(urm, *args, **kwargs)
        self.icm = icm
        self.ucm = ucm

    def fit(self, weighted=True, threshold=None):
        start_time = time.time()

        urm = self.urm * self.urm.T
        ucm = self.ucm * self.ucm.T
        urm *= ucm.sum() / urm.sum()

        W = self.alpha * urm + (1.0 - self.alpha) * ucm

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

        self._Q = -B / m  # Normalized QUBO matrix

        self._fit_time = time.time() - start_time

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        users, _ = super(QUBOProjectedCommunityDetection,
                         QUBOProjectedCommunityDetection).get_comm_from_sample(sample, n_users)
        return users, np.zeros(n_items)

    @staticmethod
    def set_alpha(alpha: float):
        QUBOProjectedCommunityDetection.alpha = alpha