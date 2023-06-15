from CommunityDetection.Communities import Communities, get_community_folder_path, CommunitiesEI
from CommunityDetection.Community import Community
from CommunityDetection.BaseCommunityDetection import BaseCommunityDetection
from CommunityDetection.QUBOBipartiteCommunityDetection import QUBOBipartiteCommunityDetection
from CommunityDetection.QUBOBipartiteProjectedCommunityDetection import QUBOBipartiteProjectedCommunityDetection
from CommunityDetection.QUBOBipartiteProjectedCommunityDetection2 import QUBOBipartiteProjectedCommunityDetection2
from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection
from CommunityDetection.CommunityDetectionRecommender import CommunityDetectionRecommender
from CommunityDetection.UserCommunityDetection import UserCommunityDetection
from CommunityDetection.HybridRecommender import HybridRecommender, calc_num_iters
from CommunityDetection.KmeansCommunityDetection import KmeansCommunityDetection
from CommunityDetection.HierarchicalClustering import HierarchicalClustering
from CommunityDetection.QUBOGraphCommunityDetection import QUBOGraphCommunityDetection
from CommunityDetection.QUBOProjectedCommunityDetection import QUBOProjectedCommunityDetection
from CommunityDetection.HybridCommunityDetection import HybridCommunityDetection
from CommunityDetection.MultiHybridCommunityDetection import MultiHybridCommunityDetection
from CommunityDetection.QUBONcutCommunityDetection import QUBONcutCommunityDetection
from CommunityDetection.SpectralClustering import SpectralClustering
from CommunityDetection.QUBOBipartiteProjectedItemCommunityDetection import QUBOBipartiteProjectedItemCommunityDetection
from CommunityDetection.LTBipartiteProjectedCommunityDetection import LTBipartiteProjectedCommunityDetection
from CommunityDetection.LTBipartiteCommunityDetection import LTBipartiteCommunityDetection
from CommunityDetection.Clusters import Clusters
from CommunityDetection.QuantityDivision import QuantityDivision
from CommunityDetection.CascadeCommunityDetection import CascadeCommunityDetection

METHOD_LIST = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection, UserCommunityDetection, \
               KmeansCommunityDetection, QUBOGraphCommunityDetection, QUBOProjectedCommunityDetection, \
               HybridCommunityDetection, MultiHybridCommunityDetection, QUBOBipartiteProjectedItemCommunityDetection, \
               SpectralClustering, LTBipartiteProjectedCommunityDetection, LTBipartiteCommunityDetection, \
               QuantityDivision, CascadeCommunityDetection]
METHOD_DICT = {method.name : method for method in METHOD_LIST}

class EmptyCommunityError(Exception):
    pass
