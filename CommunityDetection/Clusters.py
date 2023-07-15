import numpy as np

from CommunityDetection.Community import Community
from utils.DataIO import DataIO
from utils.types import List, NDArray, Optional


class Clusters:
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        self.communities_list = []
        self.num_iters = -1
    
    def add_iteration(self, clusters: List[List[int]]):
        items = np.arange(self.n_items)
        item_mask = np.ones(self.n_items).astype(bool)
        communities = []
        for cluster in clusters:
            if len(cluster) < 1:
                continue
            users = np.array(cluster)
            user_mask = np.zeros(self.n_users).astype(bool)
            user_mask[users] = True
            community = Community(users, items, user_mask, item_mask)
            communities.append(community)

        self.num_iters += 1
        self.communities_list.append(communities)
    
    def iter(self, n_iter: int = None):
        if n_iter is None:
            n_iter = self.num_iters
        assert n_iter <= self.num_iters, f'n_iter({n_iter}) >= self.num_iters({self.num_iters})'
        return self.communities_list[n_iter]