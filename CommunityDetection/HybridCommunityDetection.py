'''
Author: Kaizyn
Date: 2023-01-11 13:45:23
LastEditTime: 2023-01-13 21:18:12
'''
import time

import numpy as np

from CommunityDetection import QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection, \
    UserCommunityDetection, QUBOCommunityDetection


class HybridCommunityDetection(QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection):
    name = 'HybridCommunityDetection'

    def __init__(self, urm, icm, ucm):
        super(HybridCommunityDetection, self).__init__(urm=urm, icm=icm, ucm=ucm)

    def fit(self, *args, **kwargs):
        QUBOCommunityDetection.fit(*args, **kwargs)