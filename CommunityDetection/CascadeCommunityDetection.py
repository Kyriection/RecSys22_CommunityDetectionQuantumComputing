import logging

from CommunityDetection import QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection, \
    UserCommunityDetection, QUBOCommunityDetection, LTBipartiteProjectedCommunityDetection

logging.basicConfig(level=logging.INFO)
HYBRID_LIST = [QUBOBipartiteCommunityDetection, UserCommunityDetection]

assert len(HYBRID_LIST) >= 2

class CascadeCommunityDetection(*HYBRID_LIST):
    name = 'CascadeCommunityDetection'
    all_size = -1
    alpha = 0.5

    def __init__(self, urm, icm, ucm):
        super().__init__(urm, icm, ucm, icm, ucm)
        self.set_all_size(urm.size)
        self.filter_items = True
        self.filter_users = True
    
    def fit(self, *args, **kwargs):
        method = self.select_method(self.urm.size)
        self.filter_items = method.filter_items
        self.filter_users = method.filter_users
        method.fit(self, *args, **kwargs)

    @staticmethod
    def check_select_method(urm_size: int) -> int:
        all_size = CascadeCommunityDetection.all_size
        alpha = CascadeCommunityDetection.alpha
        if urm_size > all_size * alpha:
            return 0
        else:
            return 1

    @staticmethod
    def select_method(n_users: int) -> QUBOCommunityDetection:
        return HYBRID_LIST[CascadeCommunityDetection.check_select_method(n_users)]

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        method = CascadeCommunityDetection.select_method(n_users)
        logging.info(f'CascadeCommunityDetection select {method.name}')
        return method.get_comm_from_sample(sample, n_users, n_items=n_items)
    
    @staticmethod
    def set_alpha(alpha: float):
        CascadeCommunityDetection.alpha = alpha
    
    @staticmethod
    def set_all_size(urm_size: int):
        if CascadeCommunityDetection.all_size == -1:
            CascadeCommunityDetection.all_size = urm_size
            print(f"{CascadeCommunityDetection.name}: set all_size={urm_size}")