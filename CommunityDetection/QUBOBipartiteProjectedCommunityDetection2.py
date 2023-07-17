import time

import numpy as np

from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


def build_like_matrix(urm):
    n_items, n_users = urm.shape
    users_sum_rating = np.zeros(n_users, dtype=float)
    users_num_rating = np.zeros(n_users, dtype=int)
    rows, cols = urm.nonzero()
    for item, user in zip(rows, cols):
        users_sum_rating[user] += urm[item, user]
        users_num_rating[user] += 1
    users_num_rating[users_num_rating == 0] = 1
    users_mean_rating = users_sum_rating / users_num_rating
    for item, user in zip(rows, cols):
        urm[item, user] -= users_mean_rating[user]
    return urm


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
        urm = build_like_matrix(self.urm.copy())
        W = urm * urm.T

        if not weighted:
            W = (W > 0) * 1
        else:
            W[W < 0] = 0

        W.setdiag(0)
        W.eliminate_zeros()

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
