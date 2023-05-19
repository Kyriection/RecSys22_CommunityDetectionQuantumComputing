from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection
from CommunityDetection.QUBOBipartiteCommunityDetection import QUBOBipartiteCommunityDetection
from CommunityDetection.QUBOBipartiteProjectedCommunityDetection import QUBOBipartiteProjectedCommunityDetection
from CommunityDetection.QUBOLongTailCommunityDetection import QUBOLongTailCommunityDetection

HYBRID_LIST = [QUBOLongTailCommunityDetection, QUBOBipartiteCommunityDetection]
# HYBRID_LIST = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection]
assert len(HYBRID_LIST) >= 2

class HybridCommunityDetection2(QUBOCommunityDetection):
    name = 'HybridCommunityDetection2'
    n_all_users = -1
    alpha = 0.5

    def __init__(self, urm, icm, ucm):
        super().__init__(urm)
        self.set_n_all_users(urm.shape[0])
        self.filter_items = True
        self.filter_users = True
        self.method_list = [method(urm, icm, ucm) for method in HYBRID_LIST]
    
    def fit(self, *args, **kwargs):
        n_users = self.urm.shape[0]
        method = self.select_method(n_users)
        self.filter_items = method.filter_items
        self.filter_users = method.filter_users
        method.fit(self, *args, **kwargs)

    @staticmethod
    def check_select_method(n_users: int) -> int:
        n_all_users = HybridCommunityDetection2.n_all_users
        alpha = HybridCommunityDetection2.alpha
        if n_users > n_all_users * alpha:
            return 0
        else:
            return 1

    def select_method(self, n_users: int) -> QUBOCommunityDetection:
        return self.method_list[HybridCommunityDetection2.check_select_method(n_users)]

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        method = HybridCommunityDetection2.select_method(n_users)
        return method.get_comm_from_sample(sample, n_users, n_items=n_items)
    
    @staticmethod
    def set_alpha(alpha: float):
        HybridCommunityDetection2.alpha = alpha
    
    @staticmethod
    def set_n_all_users(n_users: int):
        if HybridCommunityDetection2.n_all_users == -1:
            HybridCommunityDetection2.n_all_users = n_users
            print(f"{HybridCommunityDetection2.name}: set n_all_users={n_users}")
