'''
Author: Kaizyn
Date: 2023-01-11 13:45:23
LastEditTime: 2023-01-13 21:18:12
'''
import time

import numpy as np
import scipy.sparse as sp

from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class QUBONcutCommunityDetection(QUBOCommunityDetection):
    filter_items = False
    name = 'QUBONcutCommunityDetection'

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(QUBONcutCommunityDetection, self).__init__(urm, *args, **kwargs)
        self.icm = icm
        self.ucm = ucm

    def fit(self, threshold = None):
        start_time = time.time()

        A = self.urm * self.urm.T
        d = A.sum(axis=0).A.flatten()
        D = sp.diags(d)
        L = A - D
        L.eliminate_zeros()
        rows, cols = L.nonzero()
        # 对拉普拉斯矩阵 L 做标准化
        for row, col in zip(rows, cols):
            L[row, col] /= np.sqrt(d[row] * d[col])

        self._Q = -L

        self._fit_time = time.time() - start_time

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        users, _ = super(QUBONcutCommunityDetection,
                         QUBONcutCommunityDetection).get_comm_from_sample(sample, n_users)
        return users, np.zeros(n_items)
