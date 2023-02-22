'''
Author: Kaizyn
Date: 2023-01-11 13:45:23
LastEditTime: 2023-01-20 23:00:04
'''
import time

import numpy as np

from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class QUBOGraphCommunityDetection(QUBOCommunityDetection):
    name = 'QUBOGraphCommunityDetection'

    def __init__(self, urm, icm, ucm):
        super(QUBOGraphCommunityDetection, self).__init__(urm=urm)
        self.icm = icm
        self.ucm = ucm

    def fit(self, threshold=None):
        start_time = time.time()

        # TODO: normalize ucm, icm and urm, or weighted
        user_block = self.ucm * self.ucm.T
        item_block = self.icm * self.icm.T
        user_block.setdiag(0)
        item_block.setdiag(0)
        user_block.eliminate_zeros()
        item_block.eliminate_zeros()
        mat = np.block([
            [user_block.toarray(), self.urm.toarray()],
            [self.urm.T.toarray(), item_block.toarray()]
        ])

        self._Q = self.get_modularity_matrix(mat)
        print(f"get_modularity_matrix():Q.shape={self._Q.shape}")

        self._fit_time = time.time() - start_time
