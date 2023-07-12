import os
import time
import os.path
import shutil
import logging
import tqdm

import argparse
import dimod
import greedy
import neal
import numpy as np
import scipy.sparse as sp
import tabu
from dwave.system import LeapHybridSampler, DWaveSampler

from CommunityDetection import BaseCommunityDetection, QUBOBipartiteCommunityDetection, \
    QUBOBipartiteProjectedCommunityDetection, Communities, CommunityDetectionRecommender, \
    get_community_folder_path, KmeansCommunityDetection, HierarchicalClustering, \
    QUBOGraphCommunityDetection, QUBOProjectedCommunityDetection, UserCommunityDetection, \
    HybridCommunityDetection, MultiHybridCommunityDetection, QUBONcutCommunityDetection, \
    SpectralClustering, QUBOBipartiteProjectedItemCommunityDetection, CommunitiesEI, \
    LTBipartiteProjectedCommunityDetection, LTBipartiteCommunityDetection, QuantityDivision, \
    METHOD_DICT, get_cascade_class, UserBipartiteCommunityDetection
from LT_qa_run_community_detection import qa_run_cd
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, \
    MovielensSampleReader, MovielensSample2Reader, MovielensSample3Reader, DATA_DICT
from recsys.Evaluation.EvaluatorSeparate import EvaluatorSeparate
from recsys.Recommenders.BaseRecommender import BaseRecommender
from recsys.Recommenders import SVRRecommender, LRRecommender, DTRecommender, RECOMMENDER_DICT
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.urm import get_community_urm, load_data, merge_sparse_matrices, head_tail_cut
from utils.plot import plot_line, plot_scatter, plot_divide, plot_metric
from utils.derived_variables import create_related_variables
from results.plot_results import print_result
from LT_qa_recommendation import load_communities, train_all_data_recommender, evaluate_recommender, \
    train_recommender_on_community, recommend_per_method
import utils.seed


logging.basicConfig(level=logging.INFO)
EVALUATE_FLAG = True


def main(data_reader_classes, method_list: Iterable[Type[BaseCommunityDetection]],
         sampler_list: Iterable[dimod.Sampler], recommender_list: Iterable[Type[BaseRecommender]],
         result_folder_path: str):
    split_quota = [80, 10, 10]
    user_wise = False
    make_implicit = False
    threshold = None

    recsys_args = {
        'cutoff_to_optimize': 10,
        # 'cutoff_list': CUTOFF_LIST,
        'n_cases': 50,
        'n_random_starts': 15,
        'metric_to_optimize': 'NDCG',
        'resume_from_saved': True,
        'similarity_type_list': ['cosine'],
    }

    save_model = True

    for data_reader_class in data_reader_classes:
        data_reader = data_reader_class()
        dataset_name = data_reader._get_dataset_name()
        urm_train, urm_validation, urm_test, icm, ucm = load_data(data_reader, split_quota=split_quota, user_wise=user_wise,
                                                        make_implicit=make_implicit, threshold=threshold, icm_ucm=True)

        urm_train, urm_validation, urm_test, icm, ucm = urm_train.T.tocsr(), urm_validation.T.tocsr(), urm_test.T.tocsr(), ucm, icm # item is main charactor
        urm_train_last_test = merge_sparse_matrices(urm_train, urm_validation)
        icm, ucm = create_related_variables(urm_train_last_test, icm, ucm) # should use urm_train_last_test ?
        icm, ucm = sp.csr_matrix(icm), sp.csr_matrix(ucm)

        for recommender in recommender_list:
            recommender_name = recommender.RECOMMENDER_NAME
            output_folder_path = f'{result_folder_path}{dataset_name}/{recommender_name}/'
            if not os.path.exists(f'{output_folder_path}baseline.zip') or not os.path.exists(
                    f'{output_folder_path}{recommender_name}_best_model_last.zip'):
                train_all_data_recommender(recommender, urm_train_last_test, urm_test, ucm, icm, dataset_name, result_folder_path)
            else:
                print(f'{recommender_name} already trained and evaluated on {dataset_name}.')

        for method in method_list:
            logging.info(f'------------start {method.name}----------')
            recommend_per_method(urm_train, urm_validation, urm_test, urm_train_last_test, ucm, icm, method, sampler_list,
                                 recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy(),
                                 save_model=save_model)


def recommend_per_method(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, sampler_list, recommender_list,
                         dataset_name, folder_path, **kwargs):
    if method.is_qubo:
        for sampler in sampler_list:
            cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name,
                              folder_path, sampler=sampler, **kwargs)
    else:
        cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name,
                          folder_path, **kwargs)


def cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name, folder_path,
                      sampler: dimod.Sampler = None, **kwargs):
    dataset_folder_path = f'{folder_path}{dataset_name}/'
    communities = load_communities(dataset_folder_path, method, sampler)
    if communities is None:
        print(f'Could not load communitites for {dataset_folder_path}, {method}, {sampler}.')
        return

    method_path = f'{dataset_folder_path}{method.name}/'
    num_iters = communities.num_iters + 1
    starting_iter = None
    sampler_name = sampler.__class__.__name__
    for n_iter in range(num_iters):
        if os.path.exists(f'{method_path}iter{n_iter:02d}/{sampler_name}_DWaveSampler/'):
            print(f'Found QPU CD at iteration {n_iter}.')
            starting_iter = n_iter
            break

    logging.info(f'starting_iter={starting_iter}')
    if starting_iter is None:
        print(f'No QPU experiments for {dataset_name} with {method} + {sampler_name}')
        return

    # if starting_iter == 0:
    #     communities = None
    # else:
    #     communities.reset_from_iter(starting_iter)

    for n_iter in range(starting_iter, num_iters):
        # new_communities = []
        # n_comm = 0
        # for community in communities.iter(n_iter):
        #     cd = qa_run_cd(cd_urm, icm, ucm, method, dataset_folder_path, base_sampler=sampler.__class__, sampler=DWaveSampler(),
        #                    community=community, n_iter=n_iter, n_comm=n_comm, **kwargs)
        #     new_communities.append(cd)
        #     n_comm += 1
        # communities.add_iteration(new_communities)

        recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name,
                           folder_path, sampler=sampler, communities=communities, n_iter=n_iter, **kwargs)


def recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name, folder_path,
                       sampler: dimod.Sampler = None, communities: Communities = None, n_iter: int = 0, **kwargs):
    method_folder_path = f'{folder_path}{dataset_name}/{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}_DWaveSampler/'

    print(f'///Training recommenders for iteration {n_iter} on {dataset_name} with {method.name} and {folder_suffix}//')

    output_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, folder_suffix=folder_suffix)
    logging.info(f'output_folder_path: {output_folder_path}')
    for recommender in recommender_list:
        recommender_name = recommender.RECOMMENDER_NAME
        print(f'{recommender_name}...')

        if not os.path.exists(f'{output_folder_path}cd_{recommender_name}.zip'):
            n_comm = 0
            cd_recommenders = []
            for community in tqdm.tqdm(communities.iter(n_iter)):
                comm_recommender = train_recommender_on_community(recommender, community, urm_train, urm_validation,
                                                                  urm_test, ucm, icm, dataset_name, folder_path,
                                                                  method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                                  folder_suffix=folder_suffix)
                cd_recommenders.append(comm_recommender)
                n_comm += 1
            evaluate_recommender(cd_urm, urm_test, ucm, icm, communities, cd_recommenders, output_folder_path,
                                 recommender_name, n_iter=n_iter)
        else:
            print('Recommender already trained and evaluated.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', nargs='+', type=str, help='method',
                        choices=['QUBOBipartiteCommunityDetection', 'QUBOBipartiteProjectedCommunityDetection',
                                 'LTBipartiteCommunityDetection', 'LTBipartiteProjectedCommunityDetection',
                                 'KmeansCommunityDetection', 'QuantityDivision', 'HybridCommunityDetection',
                                 'TestCommunityDetection'])
    parser.add_argument('-r', '--recommender', nargs='+', type=str, default=['LRRecommender'], help='recommender',
                        choices=['LRRecommender', 'SVRRecommender', 'DTRecommender'])
    parser.add_argument('-d', '--dataset', nargs='+', type=str, default=['Movielens100K'], help='dataset',
                        choices=['Movielens100K', 'Movielens1M', 'MovielensHetrec2011', 'MovielensSample',
                                 'MovielensSample2', 'MovielensSample3'])
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha for cascade')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='beta for quantity')
    parser.add_argument('-t', '--T', type=int, default=5, help='T for quantity')
    parser.add_argument('-o', '--ouput', type=str, default='results', help='the path to save the result')
    parser.add_argument('--attribute', action='store_true', help='Use item attribute data (cascade) or not')
    args = parser.parse_args()
    return args


def save_results(data_reader_classes, result_folder_path, method_list, sampler_list, recommender_list, *args):
    CUT_RATIO = 0.0
    tag = []
    for arg in args:
        if arg is None:
            continue
        tag.append(str(arg))
    tag = '_'.join(tag) if tag else '_'

    for data_reader in data_reader_classes:
        dataset_name = data_reader.DATASET_SUBFOLDER
        output_folder = os.path.join('./results/', dataset_name, 'results')
        for recommender in recommender_list:
            print_result(CUT_RATIO, data_reader, method_list, sampler_list, recommender.RECOMMENDER_NAME, False, output_folder, tag, result_folder_path)


if __name__ == '__main__':
    args = parse_args()
    # data_reader_classes = [Movielens100KReader, Movielens1MReader, FilmTrustReader, MovielensHetrec2011Reader,
                        #    LastFMHetrec2011Reader, FrappeReader, CiteULike_aReader, CiteULike_tReader]
    data_reader_classes = [DATA_DICT[data_name] for data_name in args.dataset]
    recommender_list = [RECOMMENDER_DICT[recommender_name] for recommender_name in args.recommender]
    # method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection]
    method_list = [METHOD_DICT[method_name] for method_name in args.method]
    if args.attribute:
        method_list = [get_cascade_class(method) for method in method_list]
    sampler_list = [neal.SimulatedAnnealingSampler()]
    # sampler_list = [LeapHybridSampler(), neal.SimulatedAnnealingSampler(), greedy.SteepestDescentSampler(),
                    # tabu.TabuSampler()]
    result_folder_path = f'{os.path.abspath(args.ouput)}/'
    main(data_reader_classes, method_list, sampler_list, recommender_list, result_folder_path)
    save_results(data_reader_classes, result_folder_path, method_list, sampler_list, recommender_list, args.T, args.alpha, args.beta)
