import logging
import numpy as np

from CommunityDetection import QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection, \
    UserCommunityDetection, QUBOCommunityDetection, UserBipartiteCommunityDetection, \
    LTBipartiteProjectedCommunityDetection, LTBipartiteCommunityDetection

logging.basicConfig(level=logging.INFO)
HYBRID_LIST = [QUBOBipartiteProjectedCommunityDetection, UserCommunityDetection]

assert len(HYBRID_LIST) >= 2

class TestCommunityDetection(*HYBRID_LIST):
    name = 'TestCommunityDetection'
    beta = 0.5
    filter_items = True
    filter_users = True

    def __init__(self, urm, icm, ucm):
        super().__init__(urm, icm, ucm, icm, ucm)
    
    def fit(self, *args, **kwargs):
        method = self.select_method()
        TestCommunityDetection.filter_items = method.filter_items
        TestCommunityDetection.filter_users = method.filter_users
        method.fit(self, *args, **kwargs)

    def select_method(self) -> QUBOCommunityDetection:
        size_list = [0] * len(HYBRID_LIST)
        for i, method in enumerate(HYBRID_LIST):
            if method in [QUBOBipartiteCommunityDetection, LTBipartiteCommunityDetection]:
                size_list[i] = self.urm.size
            elif method in [UserCommunityDetection]:
                size_list[i] = (self.ucm * self.ucm.T).size
            else:
                size_list[i] = (self.urm * self.urm.T).size
        info_str = f'size_list={size_list}, {round(size_list[1] / size_list[0], 2)}'
        size_list[0] *= TestCommunityDetection.beta
        method = HYBRID_LIST[np.argmax(size_list)]
        logging.info(info_str + f', use {method.name}')
        return method

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        users, items = QUBOCommunityDetection.get_comm_from_sample(sample, n_users)
        if not TestCommunityDetection.filter_items:
            items = np.zeros(n_items)
        if not TestCommunityDetection.filter_users:
            users = np.zeros(n_users)
        # logging.info(f'user {TestCommunityDetection.filter_users}, item {TestCommunityDetection.filter_items}, users {np.count_nonzero(users)}, items {np.count_nonzero(items)}')
        return users, items
    
    @staticmethod
    def set_beta(beta: float):
        TestCommunityDetection.beta = beta