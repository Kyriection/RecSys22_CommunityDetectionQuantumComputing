import time
import logging

import numpy as np

from CommunityDetection.Communities import Communities
from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class LTBipartiteCommunityDetection(QUBOCommunityDetection):
    name = 'QUBOBipartiteCommunityDetection'
    alpha: float = 0.5
    T: int = 5

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(LTBipartiteCommunityDetection, self).__init__(urm, *args, **kwargs)
        self.icm = icm
        self.ucm = ucm

    def fit(self, threshold=None):
        start_time = time.time()

        n_users, n_items = self.urm.shape

        k = self.urm.sum(axis=1)    # Degree of the user nodes
        d = self.urm.sum(axis=0)    # Degree of the item nodes
        m = k.sum()                 # Total number of graph links

        P_block = k * d / m         # Null model

        block = self.urm - P_block  # Block of the QUBO matrix

        C_quantity = np.ediff1d(self.urm.tocsr().indptr)
        C_quantity = C_quantity / np.max(C_quantity) # normalization
        diag = np.exp(C_quantity * self.T)
        diag /= np.sum(diag)
        diag -= diag.mean()
        cnt = sum(diag > 0)
        logging.info(f'{round(cnt / len(diag) * 100, 2)}%({cnt}) get benefit from C_quantity.')
        diag *= (1 - self.alpha)
        block *= self.alpha / np.max(np.abs(block))
        block_user = np.zeros((n_users, n_users))
        np.fill_diagonal(block_user, diag)

        B = np.block([
            # [np.zeros((n_users, n_users)), block],
            [block_user, block],
            [block.T, np.zeros((n_items, n_items))]
        ])

        if threshold is not None:
            B[np.abs(B) < threshold] = 0

        # self._Q = -B / m            # Normalized QUBO matrix
        self._Q = -B

        self._fit_time = time.time() - start_time
    
    @staticmethod
    def set_alpha(alpha: float):
        LTBipartiteCommunityDetection.alpha = alpha
    
    @staticmethod
    def set_T(T: int):
        LTBipartiteCommunityDetection.T = T
    