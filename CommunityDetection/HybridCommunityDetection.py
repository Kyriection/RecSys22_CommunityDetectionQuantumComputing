import logging

from CommunityDetection import QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection, \
    UserCommunityDetection, QUBOCommunityDetection, LTBipartiteProjectedCommunityDetection, LTBipartiteCommunityDetection

logging.basicConfig(level=logging.INFO)
HYBRID_LIST = [LTBipartiteCommunityDetection, QUBOBipartiteCommunityDetection]

assert len(HYBRID_LIST) >= 2

class HybridCommunityDetection(*HYBRID_LIST):
    name = 'HybridCommunityDetection'
    n_all_users = -1
    beta = 0.5

    def __init__(self, urm, icm, ucm):
        super().__init__(urm, icm, ucm, icm, ucm)
        self.set_n_all_users(urm.shape[0])
        self.filter_items = True
        self.filter_users = True
    
    def fit(self, *args, **kwargs):
        n_users = self.urm.shape[0]
        method = self.select_method(n_users)
        self.filter_items = method.filter_items
        self.filter_users = method.filter_users
        method.fit(self, *args, **kwargs)

    @staticmethod
    def check_select_method(n_users: int) -> int:
        n_all_users = HybridCommunityDetection.n_all_users
        beta = HybridCommunityDetection.beta
        if n_users > n_all_users * beta:
            return 0
        else:
            return 1

    @staticmethod
    def select_method(n_users: int) -> QUBOCommunityDetection:
        return HYBRID_LIST[HybridCommunityDetection.check_select_method(n_users)]

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        method = HybridCommunityDetection.select_method(n_users)
        logging.info(f'use {method.name}')
        return method.get_comm_from_sample(sample, n_users, n_items=n_items)
    
    @staticmethod
    def set_beta(beta: float):
        HybridCommunityDetection.beta = beta
    
    @staticmethod
    def set_n_all_users(n_users: int):
        if HybridCommunityDetection.n_all_users == -1:
            HybridCommunityDetection.n_all_users = n_users
            print(f"{HybridCommunityDetection.name}: set n_all_users={n_users}")
    
    def get_graph_cut(self, *args):
        pass