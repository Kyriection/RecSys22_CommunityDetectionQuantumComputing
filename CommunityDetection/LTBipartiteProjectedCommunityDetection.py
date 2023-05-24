import logging
import time

import numpy as np

from CommunityDetection.Communities import Communities
from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection

logging.basicConfig(level=logging.INFO)


class LTBipartiteProjectedCommunityDetection(QUBOCommunityDetection):
    filter_items = False
    name = 'QUBOBipartiteProjectedCommunityDetection'
    alpha = 0.5

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(LTBipartiteProjectedCommunityDetection, self).__init__(urm, *args, **kwargs)
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
        T = 12
        diag = np.exp(C_quantity / T)
        diag /= np.sum(diag)
        diag = (diag - diag.mean())
        # logging.info(f'B_max={np.max(B)}, diag_max={np.max(diag)}')
        B *= self.alpha
        for i in range(len(diag)):
            B[i, i] += (1 - self.alpha) * diag[i]

        if threshold is not None:
            B[np.abs(B) < threshold] = 0

        self.W = W
        self._Q = -B / m  # Normalized QUBO matrix

        self._fit_time = time.time() - start_time

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        users, _ = super(LTBipartiteProjectedCommunityDetection,
                         LTBipartiteProjectedCommunityDetection).get_comm_from_sample(sample, n_users)
        return users, np.zeros(n_items)

    @staticmethod
    def set_alpha(alpha: float):
        LTBipartiteProjectedCommunityDetection.alpha = alpha