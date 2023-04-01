'''
Author: Kaizyn
Date: 2023-01-11 13:45:23
LastEditTime: 2023-01-20 23:00:04
'''
import time

import numpy as np

from CommunityDetection.Communities import Communities
from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class QUBOBipartiteCommunityDetection(QUBOCommunityDetection):
    name = 'QUBOBipartiteCommunityDetection'

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(QUBOBipartiteCommunityDetection, self).__init__(urm, *args, **kwargs)
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
    
    def get_graph_cut(self, communities: Communities):
        cut = 0.0
        rows, cols = self.urm.nonzero()
        for row, col in zip(rows, cols):
            if communities.user_mask[row] != communities.item_mask[col]:
                cut += self.urm[row, col]
        all = self.urm.sum()
        # print(f'cut/all={cut}/{all}={cut/all}')
        return cut, all
    