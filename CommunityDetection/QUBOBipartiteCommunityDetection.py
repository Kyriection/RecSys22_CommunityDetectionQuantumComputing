'''
Author: Kaizyn
Date: 2023-01-11 13:45:23
LastEditTime: 2023-01-20 23:00:04
'''
import time

import numpy as np

from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class QUBOBipartiteCommunityDetection(QUBOCommunityDetection):
    name = 'QUBOBipartiteCommunityDetection'

    def __init__(self, urm, icm, ucm):
        super(QUBOBipartiteCommunityDetection, self).__init__(urm=urm)
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

        B = np.block([
            [np.zeros((n_users, n_users)), block],
            [block.T, np.zeros((n_items, n_items))]
        ])

        if threshold is not None:
            B[np.abs(B) < threshold] = 0

        self._Q = -B / m            # Normalized QUBO matrix

        self._fit_time = time.time() - start_time
    """
    def fit(self, threshold=None):
        start_time = time.time()

        n_users, n_items = self.urm.shape

        def calc_block(mat):
            k = mat.sum(axis=1)
            d = mat.sum(axis=0)
            m = k.sum()
            P_block = k * d / m
            block = mat - P_block
            return block, m
        
        ur_block, ur_m = calc_block(self.urm)
        uc_block, uc_m = calc_block(self.ucm * self.ucm.T)
        ic_block, ic_m = calc_block(self.icm * self.icm.T)

        B = np.block([
            [uc_block / uc_m, ur_block / ur_m],
            [ur_block.T / ur_m, ic_block / ic_m]
        ])

        if threshold is not None:
            B[np.abs(B) < threshold * uc_m] = 0

        self._Q = -B

        self._fit_time = time.time() - start_time
    """
