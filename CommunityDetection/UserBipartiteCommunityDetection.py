import time

import numpy as np

from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class UserBipartiteCommunityDetection(QUBOCommunityDetection):
    name = 'UserBipartiteCommunityDetection'

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(UserBipartiteCommunityDetection, self).__init__(urm, *args, **kwargs)
        self.ucm = ucm

    def fit(self, threshold=None):
        start_time = time.time()

        n_users, n_feats = self.ucm.shape

        k = self.ucm.sum(axis=1)    # Degree of the user nodes
        d = self.ucm.sum(axis=0)    # Degree of the item nodes
        m = k.sum()                 # Total number of graph links

        P_block = k * d / m         # Null model

        block = self.ucm - P_block  # Block of the QUBO matrix

        B = np.block([
            [np.zeros((n_users, n_users)), block],
            [block.T, np.zeros((n_feats, n_feats))]
        ])

        if threshold is not None:
            B[np.abs(B) < threshold] = 0

        self._Q = -B / m            # Normalized QUBO matrix

        self._fit_time = time.time() - start_time
    