from CommunityDetection import QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection, \
    UserCommunityDetection, QUBOCommunityDetection

HYBRID_LIST = [QUBOBipartiteCommunityDetection, UserCommunityDetection, QUBOBipartiteCommunityDetection]
assert len(HYBRID_LIST) >= 3

class MultiHybridCommunityDetection(*list(set(HYBRID_LIST))):
    name = 'MultiHybridCommunityDetection'
    n_all_users = -1
    alpha = 0.5
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
        n_all_users = MultiHybridCommunityDetection.n_all_users
        alpha = MultiHybridCommunityDetection.alpha
        beta = MultiHybridCommunityDetection.beta
        assert alpha >= beta
        if n_users >= n_all_users * alpha:
            return 0
        elif n_users >= n_all_users * beta:
            return 1
        else:
            return 2

    @staticmethod
    def select_method(n_users: int) -> QUBOCommunityDetection:
        return HYBRID_LIST[MultiHybridCommunityDetection.check_select_method(n_users)]

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        method = MultiHybridCommunityDetection.select_method(n_users)
        return method.get_comm_from_sample(sample, n_users, n_items=n_items)
    
    @staticmethod
    def set_alpha(alpha: float):
        MultiHybridCommunityDetection.alpha = alpha

    @staticmethod
    def set_beta(beta: float):
        MultiHybridCommunityDetection.beta = beta
    
    @staticmethod
    def set_n_all_users(n_users: int):
        if MultiHybridCommunityDetection.n_all_users == -1:
            MultiHybridCommunityDetection.n_all_users = n_users
            print(f"{MultiHybridCommunityDetection.name}: set n_all_users={n_users}")