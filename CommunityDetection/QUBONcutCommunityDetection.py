import time

import numpy as np
import scipy.sparse as sp

from CommunityDetection.Communities import Communities
from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class QUBONcutCommunityDetection(QUBOCommunityDetection):
    filter_items = False
    name = 'QUBONcutCommunityDetection'

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        super(QUBONcutCommunityDetection, self).__init__(urm, *args, **kwargs)
        self.icm = icm
        self.ucm = ucm
        self.W = None

    def fit(self, threshold = None):
        start_time = time.time()

        A = self.urm * self.urm.T
        d = A.sum(axis=0).A.flatten()
        D = sp.diags(d)
        L = D - A
        L.eliminate_zeros()
        # 对拉普拉斯矩阵 L 做标准化
        rows, cols = L.nonzero()
        for row, col in zip(rows, cols):
            if d[row] * d[col] > 0:
                L[row, col] /= np.sqrt(d[row] * d[col])

        self.W = A
        self._Q = -L.todense()

        self._fit_time = time.time() - start_time

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        users, _ = super(QUBONcutCommunityDetection,
                         QUBONcutCommunityDetection).get_comm_from_sample(sample, n_users)
        return users, np.zeros(n_items)

    def get_graph_cut(self, communities: Communities):
        cut = 0.0
        rows, cols = self.W.nonzero()
        for row, col in zip(rows, cols):
            if communities.user_mask[row] != communities.user_mask[col]:
                cut += self.W[row, col]
        all = self.W.sum()
        print(f'cut/all={cut}/{all}={cut/all}')
        return cut / all