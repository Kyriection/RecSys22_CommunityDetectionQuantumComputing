import logging

from CommunityDetection.BaseCommunityDetection import BaseCommunityDetection
from CommunityDetection.UserCommunityDetection import UserCommunityDetection
from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection

def get_cascade_class(method: BaseCommunityDetection):
    method_list = [method, UserCommunityDetection]
    class CascadeCommunityDetection(*method_list):
        name = f'Cascade-{method.name}'
        n_all_users = -1
        beta = 0.5

        def __init__(self, urm, icm, ucm, *args):
            super().__init__(urm, icm, ucm, icm, ucm)
            self.set_n_all_users(urm.shape[0])
            n_users = urm.shape[0]
            self.method = self.select_method(n_users)
            self.filter_items = self.method.filter_items
            self.filter_users = self.method.filter_users

        def fit(self, *args, **kwargs):
            self.method.fit(self, *args, **kwargs)

        @staticmethod
        def check_select_method(n_users: int) -> int:
            n_all_users = CascadeCommunityDetection.n_all_users
            beta = CascadeCommunityDetection.beta
            if n_users > n_all_users * beta:
                return 0
            else:
                return 1

        @staticmethod
        def select_method(n_users: int) -> QUBOCommunityDetection:
            return method_list[CascadeCommunityDetection.check_select_method(n_users)]

        @staticmethod
        def get_comm_from_sample(sample, n_users, n_items=0):
            method = CascadeCommunityDetection.select_method(n_users)
            logging.info(f'n_users={n_users}, use {method.name}')
            return method.get_comm_from_sample(sample, n_users, n_items=n_items)

        @staticmethod
        def set_beta(beta: float):
            CascadeCommunityDetection.beta = beta

        @staticmethod
        def set_n_all_users(n_users: int):
            if CascadeCommunityDetection.n_all_users == -1:
                CascadeCommunityDetection.n_all_users = n_users
                logging.info(f"{CascadeCommunityDetection.name}: set n_all_users={n_users}")

    return CascadeCommunityDetection