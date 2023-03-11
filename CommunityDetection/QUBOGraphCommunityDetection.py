'''
Author: Kaizyn
Date: 2023-01-11 13:45:23
LastEditTime: 2023-01-20 23:00:04
'''
import time

import scipy.sparse as sp

from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class QUBOGraphCommunityDetection(QUBOCommunityDetection):
    name = 'QUBOGraphCommunityDetection'
    alpha = 0.5

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(QUBOGraphCommunityDetection, self).__init__(urm, *args, **kwargs)
        self.icm = icm
        self.ucm = ucm

    def fit(self, threshold=None):
        start_time = time.time()

        user_block = self.ucm * self.ucm.T
        item_block = self.icm * self.icm.T
        user_block.setdiag(0)
        item_block.setdiag(0)
        user_block.eliminate_zeros()
        item_block.eliminate_zeros()
        urm_sum = self.urm.sum()
        user_block *= urm_sum / user_block.sum()
        item_block *= urm_sum / item_block.sum()
        mat = sp.bmat([
            [(1 - self.alpha) * user_block, self.alpha * self.urm],
            [self.alpha * self.urm.T, (1 - self.alpha) * item_block]
        ])

        self._Q = self.get_modularity_matrix(mat)
        print(f"get_modularity_matrix():Q.shape={self._Q.shape}")

        self._fit_time = time.time() - start_time

    @staticmethod
    def set_alpha(alpha: float):
        QUBOGraphCommunityDetection.alpha = alpha