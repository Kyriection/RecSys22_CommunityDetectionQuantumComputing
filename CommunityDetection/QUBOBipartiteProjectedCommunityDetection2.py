import time

import numpy as np

from CommunityDetection.Communities import Communities
from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class QUBOBipartiteProjectedCommunityDetection2(QUBOCommunityDetection):
    filter_items = False
    name = 'QUBOBipartiteProjectedCommunityDetection2'

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(QUBOBipartiteProjectedCommunityDetection2, self).__init__(urm, *args, **kwargs)
        self.icm = icm
        self.ucm = ucm
        self.W = None

    def fit(self, weighted=True, threshold=None):
        start_time = time.time()

        # W = self.urm * self.urm.T

        # rebuild urm
        urm = (self.urm - self.urm.mean(axis=1)) / self.urm.max()
        W = urm * urm.T
        min_val = np.min(W)
        max_val = np.max(W)
        W = (W - min_val) / (max_val - min_val)

        if not weighted:
            W = (W > 0) * 1

        # W.setdiag(0)
        # W.eliminate_zeros()
        W[np.diag_indices_from(W)] = 0

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
    def get_comm_from_sample(sample, n_users, n_items=0):
        users, _ = super(QUBOBipartiteProjectedCommunityDetection2,
                         QUBOBipartiteProjectedCommunityDetection2).get_comm_from_sample(sample, n_users)
        return users, np.zeros(n_items)
