import os
import time
import copy
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
from dwave.system import LeapHybridSampler
from sklearn.cluster import KMeans, SpectralClustering

from CommunityDetection import BaseCommunityDetection, QUBOBipartiteCommunityDetection, \
    QUBOBipartiteProjectedCommunityDetection, Communities, CommunityDetectionRecommender, \
    get_community_folder_path, KmeansCommunityDetection, HierarchicalClustering, \
    QUBOGraphCommunityDetection, QUBOProjectedCommunityDetection, UserCommunityDetection, \
    HybridCommunityDetection, MultiHybridCommunityDetection, QUBONcutCommunityDetection, \
    SpectralClustering, QUBOBipartiteProjectedItemCommunityDetection, CommunitiesEI, \
    LTBipartiteProjectedCommunityDetection, LTBipartiteCommunityDetection, QuantityDivision, \
    METHOD_DICT, get_cascade_class, UserBipartiteCommunityDetection, Clusters, EachItem
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, \
    MovielensSampleReader, MovielensSample2Reader, MovielensSample3Reader, DATA_DICT
from recsys.Evaluation.EvaluatorSeparate import EvaluatorSeparate
from recsys.Recommenders.BaseRecommender import BaseRecommender
from recsys.Recommenders import SVRRecommender, LRRecommender, DTRecommender, RECOMMENDER_DICT
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.urm import get_community_urm, load_data_k_fold, merge_sparse_matrices, head_tail_cut_k_fold
from utils.plot import plot_line, plot_scatter, plot_divide, plot_metric
from utils.derived_variables import create_related_variables
from results.plot_results import print_result_, print_result_k_fold, print_result_k_fold_mean
import utils.seed


logging.basicConfig(level=logging.INFO)
CUT_RATIO: float = None
EI: bool = False # EI if True else (TC or CT)
MIN_RATINGS_PER_USER = 1
EVALUATE_FLAG = False
N_CLUSTER = [2**(i+1) for i in range(10)]


def load_classical_communities(urm, ucm, method):
    n_users, n_items = urm.shape
    communities = Clusters(n_users, n_items)
    X = urm
    if 'Cascade' in method.name:
        X = sp.hstack((urm, ucm))
    for n_clusters in tqdm.tqdm(N_CLUSTER, desc='load_communities'):
        if n_clusters >= n_users:
            break
        clusters = [[] for i in range(n_clusters)]
        model = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        # model = SpectralClustering(n_clusters=n_clusters, random_state=0).fit(X)
        # model = SpectralClustering(
        #     n_clusters=n_clusters,
        #     eigen_solver='arpack',
        #     # eigen_solver='lobpcg',
        #     # eigen_solver='amg',
        #     random_state=0,
        #     assign_labels='discretize',
        #     # affinity = 'precomputed', 
        #     # n_init=1000,
        # ).fit(X)
        for i, cluster in enumerate(model.labels_):
            clusters[cluster].append(i)
        communities.add_iteration(clusters)
        cnt = sum([1 if len(cluster) > 0 else 0 for cluster in clusters])
        logging.info(f'n_clusters = {n_clusters}, cnt = {cnt}.')
    
    return communities


def load_communities(folder_path, method, sampler=None, n_iter=0, n_comm=None):
    method_folder_path = f'{folder_path}{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}/'

    try:
        communities = Communities.load(method_folder_path, 'communities', n_iter=n_iter, n_comm=n_comm,
                                       folder_suffix=folder_suffix)
        print(f'Loaded previously computed communities for {communities.num_iters + 1} iterations.')
    except FileNotFoundError:
        print('No communities found to load. Computing new communities...')
        communities = None
    return communities


def train_all_data_recommender(recommender: Type[BaseRecommender], urm_train_last_test, urm_test, ucm, icm, dataset_name: str,
                               result_folder_path: str):
    recommender_name = recommender.RECOMMENDER_NAME
    output_folder_path = f'{result_folder_path}{dataset_name}/{recommender_name}/'

    evaluator_test = EvaluatorSeparate(urm_test)

    print(f'Training {recommender_name} on all data...')

    time_on_train = time.time()
    rec = recommender(urm_train_last_test, ucm, icm)
    rec.fit()
    time_on_train = time.time() - time_on_train

    if os.path.exists(output_folder_path):
        return

    time_on_test = time.time()
    result_df, result_string = evaluator_test.evaluateRecommender(rec)
    time_on_test = time.time() - time_on_test

    data_dict_to_save = {
        'result_df': result_df,
        'result_string': result_string,
        'time_on_train': time_on_train,
        'time_on_test': time_on_test,
    }

    output_dataIO = DataIO(output_folder_path)
    output_dataIO.save_data('baseline', data_dict_to_save)

    rec.save_model(output_folder_path, f'{recommender_name}_best_model_last')


def train_recommender_on_community(recommender, community, urm_train, urm_test, ucm, icm, dataset_name,
                                   result_folder_path, method_folder_path, n_iter=0, n_comm=None, folder_suffix='',
                                   **kwargs):
    recommender_name = recommender.RECOMMENDER_NAME
    print(f'Training {recommender_name} on community {n_comm if n_comm is not None else ""} of iteration {n_iter}...')
    logging.info(f"community.size({len(community.users)}, {len(community.items)})")
    logging.info(f"community.shape({len(community.user_mask)}, {len(community.item_mask)})")
    logging.debug(f'urm.shape: {urm_train.shape}')

    output_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                   folder_suffix=folder_suffix)
    output_folder_path = f'{output_folder_path}{recommender_name}/'

    base_recommender_path = f'{result_folder_path}{dataset_name}/{recommender_name}/'

    c_urm_train, _, _, c_icm, c_ucm = get_community_urm(urm_train, community=community, filter_items=False, icm=icm, ucm=ucm)
    c_urm_test, _, _ = get_community_urm(urm_test, community=community, filter_items=False)

    ignore_users = np.arange(c_urm_train.shape[0])[np.logical_not(community.user_mask)]
    evaluator_test = EvaluatorSeparate(c_urm_test, ignore_users=ignore_users)

    time_on_train = time.time()
    comm_recommender = recommender(c_urm_train, c_ucm, c_icm, community.users)
    comm_recommender.fit()
    time_on_train = time.time() - time_on_train

    if EVALUATE_FLAG:
        time_on_test = time.time()
        result_df, result_string = evaluator_test.evaluateRecommender(comm_recommender)
        time_on_test = time.time() - time_on_test

        data_dict_to_save = {
            'result_df': result_df,
            'time_on_train': time_on_train,
            'time_on_test': time_on_test,
        }

        output_dataIO = DataIO(output_folder_path)
        output_dataIO.save_data('test', data_dict_to_save)
        community.result_dict_test = copy.deepcopy(data_dict_to_save)

        print(f'Evaluating base model on community {n_comm if n_comm is not None else ""} of iteration {n_iter}...')
        base_recommender = recommender(c_urm_train, c_ucm, c_icm)
        recommender_file_name = f'{recommender_name}_best_model_last'
        base_recommender.load_model(base_recommender_path, recommender_file_name)
        base_evaluator_test = EvaluatorSeparate(c_urm_test, ignore_users=ignore_users)

        time_on_test = time.time()
        result_df, result_string = base_evaluator_test.evaluateRecommender(base_recommender)
        time_on_test = time.time() - time_on_test

        baseline_dict = {
            'result_df': result_df,
            'time_on_test': time_on_test,
        }
        output_dataIO.save_data('baseline', baseline_dict)

    return comm_recommender


def evaluate_recommender(urm_train_last_test, urm_test, ucm, icm, communities, recommenders, output_folder_path=None, recommender_name=None,
                         n_iter=None, min_ratings_per_user: int = 1, ignore_users=None):
    print(f'Evaluating {recommender_name} on the result of community detection.')

    recommender = CommunityDetectionRecommender(urm_train_last_test, communities=communities, recommenders=recommenders,
                                                n_iter=n_iter)
    # urm_all = merge_sparse_matrices(urm_train_last_test, urm_test)
    # evaluator_test = EvaluatorSeparate(urm_all, min_ratings_per_user=min_ratings_per_user, ignore_users=ignore_users)
    evaluator_test = EvaluatorSeparate(urm_test, min_ratings_per_user=min_ratings_per_user, ignore_users=ignore_users)
    time_on_test = time.time()
    result_df, result_string = evaluator_test.evaluateRecommender(recommender)
    time_on_test = time.time() - time_on_test

    result_dict = {
        'result_df': result_df,
        'time_on_test': time_on_test,
    }
    
    if output_folder_path is not None:
        dataIO = DataIO(output_folder_path)
        dataIO.save_data(f'cd_{recommender_name}', result_dict)

    return result_dict


def main(data_reader_classes, method_list: Iterable[Type[BaseCommunityDetection]],
         sampler_list: Iterable[dimod.Sampler], recommender_list: Iterable[Type[BaseRecommender]],
         results_folder_path: str, n_folds: int, *args):
    global EI, CUT_RATIO
    user_wise = False
    make_implicit = False
    threshold = None
    tag = 'recommend'

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
        C_quantity = None
        for k in range(n_folds):
            result_folder_path = f'{results_folder_path}fold-{k:02d}/'

            urm_train, urm_test, icm, ucm = load_data_k_fold(tag, data_reader, user_wise=user_wise,make_implicit=make_implicit,
                                                            threshold=threshold, icm_ucm=True, n_folds=n_folds, k=k)

            # item is main charactor
            urm_train, urm_test, icm, ucm = urm_train.T.tocsr(), urm_test.T.tocsr(), ucm, icm
            icm, ucm = create_related_variables(urm_train, icm, ucm)
            icm, ucm = sp.csr_matrix(icm), sp.csr_matrix(ucm)

            if C_quantity is None: C_quantity = np.zeros(urm_train.shape[0])
            C_quantity += np.ediff1d(urm_train.tocsr().indptr) + np.ediff1d(urm_test.tocsr().indptr)

            h_urm_train, h_urm_test, h_icm, h_ucm,\
            t_urm_train, t_urm_test, t_icm, t_ucm = \
                head_tail_cut_k_fold(CUT_RATIO, urm_train, urm_test, icm, ucm)
            head_flag = h_urm_train.shape[0] > 0
            logging.info(f'head shape: {h_urm_train.shape}, tail shape: {t_urm_train.shape}')

            for recommender in recommender_list:
                recommender.set_limit(min(*urm_train.data, *urm_test.data), max(*urm_train.data, *urm_test.data))
                recommender_name = recommender.RECOMMENDER_NAME
                output_folder_path = f'{result_folder_path}{dataset_name}/{recommender_name}/'
                if not os.path.exists(f'{output_folder_path}baseline.zip') or not os.path.exists(
                        f'{output_folder_path}{recommender_name}_best_model_last.zip'):
                    train_all_data_recommender(recommender, t_urm_train, t_urm_test, t_ucm, t_icm, dataset_name, result_folder_path)
                else:
                    print(f'{recommender_name} already trained and evaluated on {dataset_name}.')

            for method in method_list:
                logging.info(f'------------start {method.name}----------')
                recommend_per_method(t_urm_train, t_urm_test, t_ucm, t_icm, method, sampler_list,
                                    recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy,
                                    save_model=save_model, each_item=EI)
                if not head_flag or EI:
                    continue
                recommend_per_method(h_urm_train, h_urm_test, h_ucm, h_icm, method, sampler_list,
                                    recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy,
                                    save_model=save_model, each_item=True)

    save_results(data_reader_classes, results_folder_path, method_list, sampler_list, recommender_list, n_folds, C_quantity, *args)


def recommend_per_method(urm_train, urm_test, ucm, icm, method, sampler_list, recommender_list,
                         dataset_name, folder_path, **kwargs):
    if method.is_qubo:
        for sampler in sampler_list:
            cd_recommendation(urm_train, urm_test, ucm, icm, method, recommender_list, dataset_name,
                              folder_path, sampler=sampler, **kwargs)
    else:
        cd_recommendation(urm_train, urm_test, ucm, icm, method, recommender_list, dataset_name,
                          folder_path, **kwargs)


def cd_recommendation(urm_train, urm_test, ucm, icm, method, recommender_list, dataset_name, folder_path,
                      sampler: dimod.Sampler = None, each_item: bool = False, **kwargs):
    if each_item:
        n_users, n_items = urm_train.shape
        recommend_per_iter(urm_train, urm_test, ucm, icm, method, recommender_list, dataset_name,
                           folder_path, sampler=sampler, communities=CommunitiesEI(n_users, n_items), n_iter=-1, **kwargs)
        return

    if 'KmeansCommunityDetection' in method.name:
        communities = load_classical_communities(urm_train, ucm, method)
    else:
        dataset_folder_path = f'{folder_path}{dataset_name}/'
        communities = load_communities(dataset_folder_path, method, sampler)
        if communities is None:
            print(f'Could not load communitites for {dataset_folder_path}, {method}, {sampler}.')
            return

    num_iters = communities.num_iters + 1
    for n_iter in range(num_iters):
        recommend_per_iter(urm_train, urm_test, ucm, icm, method, recommender_list, dataset_name,
                           folder_path, sampler=sampler, communities=communities, n_iter=n_iter, **kwargs)
    # plot_metric(communities, method_folder_path)


def recommend_per_iter(urm_train, urm_test, ucm, icm, method, recommender_list, dataset_name, folder_path,
                       sampler: dimod.Sampler = None, communities: Communities = None, n_iter: int = 0, **kwargs):
    method_folder_path = f'{folder_path}{dataset_name}/{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}/'

    print(f'///Training recommenders for iteration {n_iter} on {dataset_name} with {method.name} and {folder_suffix}//')

    output_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, folder_suffix=folder_suffix)
    for recommender in recommender_list:
        recommender_name = recommender.RECOMMENDER_NAME
        print(f'{recommender_name}...')

        if not os.path.exists(f'{output_folder_path}cd_{recommender_name}.zip'):
            n_comm = 0
            cd_recommenders = []
            for community in tqdm.tqdm(communities.iter(n_iter)):
                comm_recommender = train_recommender_on_community(recommender, community, urm_train,
                                                                  urm_test, ucm, icm, dataset_name, folder_path,
                                                                  method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                                  folder_suffix=folder_suffix)
                cd_recommenders.append(comm_recommender)
                n_comm += 1
            evaluate_recommender(urm_train, urm_test, ucm, icm, communities, cd_recommenders, output_folder_path,
                                 recommender_name, n_iter=n_iter)
            open(f'{output_folder_path}C{n_comm}', 'a').close()
        else:
            print('Recommender already trained and evaluated.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', nargs='+', type=str, help='method',
                        choices=['QUBOBipartiteCommunityDetection', 'QUBOBipartiteProjectedCommunityDetection',
                                 'LTBipartiteCommunityDetection', 'LTBipartiteProjectedCommunityDetection',
                                 'KmeansCommunityDetection', 'QuantityDivision', 'HybridCommunityDetection',
                                 'KmeansBipartiteCommunityDetection', 'EachItem', 'TestCommunityDetection',
                                 'QUBOBipartiteProjectedCommunityDetection2'])
    parser.add_argument('-r', '--recommender', nargs='+', type=str, default=['LRRecommender'], help='recommender',
                        choices=['LRRecommender', 'SVRRecommender', 'DTRecommender'])
    parser.add_argument('-d', '--dataset', nargs='+', type=str, default=['Movielens100K'], help='dataset',
                        choices=['Movielens100K', 'Movielens1M', 'MovielensHetrec2011', 'MovielensSample',
                                 'MovielensSample2', 'MovielensSample3', 'MovielensLatestSmall'])
    parser.add_argument('-c', '--cut_ratio', type=float, default=0.0, help='head ratio for clustered tail')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha for cascade')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='beta for quantity')
    parser.add_argument('-t', '--T', type=int, default=5, help='T for quantity')
    parser.add_argument('-l', '--layer', type=int, default=0, help='number of layer of quantity')
    parser.add_argument('-o', '--ouput', type=str, default='results', help='the path to save the result')
    parser.add_argument('-k', '--kfolds', type=int, default=5, help='number of folds for dataset split.')
    parser.add_argument('--attribute', action='store_true', help='Use item attribute data (cascade) or not')
    parser.add_argument('--implicit', action='store_true', help='URM make implicit (values to 0/1) or not.')
    parser.add_argument('--EI', action='store_true', help='Each Item')
    args = parser.parse_args()
    return args


def clean_results(results_folder_path, data_reader_classes, method_list, sampler_list, recommender_list, n_folds: int = 5):
    for k in range(n_folds):
        result_folder_path = f'{results_folder_path}fold-{k:02d}/'
        for data_reader_class in data_reader_classes:
            data_reader = data_reader_class()
            dataset_name = data_reader._get_dataset_name()
            dataset_folder_path = f'{result_folder_path}{dataset_name}/'
            if not os.path.exists(dataset_folder_path):
                continue
            hybrid_folder_path = os.path.join(dataset_folder_path, 'Hybrid')
            logging.debug(f'clean {hybrid_folder_path}')
            if os.path.exists(hybrid_folder_path):
                shutil.rmtree(hybrid_folder_path)
            for recommender in recommender_list:
                recommender_folder_path = os.path.join(dataset_folder_path, recommender.RECOMMENDER_NAME)
                logging.debug(f'clean {recommender_folder_path}')
                if os.path.exists(recommender_folder_path):
                    shutil.rmtree(recommender_folder_path)
            for method in method_list:
                method_folder_path = f'{dataset_folder_path}{method.name}/'
                if not os.path.exists(method_folder_path):
                    continue
                # print('in: ', method_folder_path)
                for iter in os.listdir(method_folder_path):
                    iter_folder_path = os.path.join(method_folder_path, iter)
                    if not os.path.isdir(iter_folder_path) or len(iter) < 4 or iter[:4] != 'iter':
                        continue
                    # print('in: ', iter_folder_path)
                    for sampler in sampler_list:
                        # sampler_folder_path = os.path.join(iter_folder_path, 'SimulatedAnnealingSampler')
                        sampler_folder_path = os.path.join(iter_folder_path, sampler.__class__.__name__)
                        if not os.path.exists(sampler_folder_path):
                            continue
                        # print('in: ', sampler_folder_path)
                        for recommender in recommender_list:
                            result_file = os.path.join(sampler_folder_path, f'cd_{recommender.RECOMMENDER_NAME}.zip')
                            logging.debug(f'clean {result_file}')
                            if os.path.exists(result_file):
                                # print('remove: ', result_file)
                                os.remove(result_file)
                            for c in os.listdir(sampler_folder_path):
                                c_folder_path = os.path.join(sampler_folder_path, c)
                                logging.debug(f'clean {c_folder_path}')
                                if os.path.isdir(c_folder_path) and c[0] == 'c':
                                    # print('remove: ', c_folder_path)
                                    shutil.rmtree(c_folder_path)


def save_results(data_reader_classes, results_folder_path, method_list, sampler_list, recommender_list, n_folds, C_quantity, *args):
    global CUT_RATIO
    tag = []
    for arg in args:
        if arg is None:
            continue
        tag.append(str(arg))
    tag = '_'.join(tag) if tag else '_'

    for data_reader in data_reader_classes:
        dataset_name = data_reader.DATASET_SUBFOLDER
        # output_folder = os.path.join('./results/', dataset_name, 'results')
        output_folder = None
        for recommender in recommender_list:
            for k in range(n_folds):
                result_folder_path = f'{results_folder_path}fold-{k:02d}/'
                print_result_(C_quantity, CUT_RATIO, data_reader, method_list, sampler_list,
                              recommender.RECOMMENDER_NAME, False, output_folder, tag, result_folder_path)
        print_result_k_fold(C_quantity, CUT_RATIO, data_reader, method_list, sampler_list,
                            recommender.RECOMMENDER_NAME, False, output_folder, tag, results_folder_path, n_folds)
        # print_result_k_fold_mean(data_reader, method_list, sampler_list, recommender.RECOMMENDER_NAME, output_folder, tag, results_folder_path, n_folds)


if __name__ == '__main__':
    args = parse_args()
    EI = args.EI
    CUT_RATIO = args.cut_ratio
    data_reader_classes = [DATA_DICT[data_name] for data_name in args.dataset]
    # data_reader_classes = [Movielens100KReader, Movielens1MReader, FilmTrustReader, MovielensHetrec2011Reader,
                        #    LastFMHetrec2011Reader, FrappeReader, CiteULike_aReader, CiteULike_tReader]
    # recommender_list = [LRRecommender]
    # recommender_list = [SVRRecommender]
    recommender_list = [RECOMMENDER_DICT[recommender_name] for recommender_name in args.recommender]
    method_list = [METHOD_DICT[method_name] for method_name in args.method]
    if args.attribute:
        method_list = [get_cascade_class(method) for method in method_list]
    sampler_list = [neal.SimulatedAnnealingSampler()]
    # sampler_list = [greedy.SteepestDescentSampler(), tabu.TabuSampler()]
    # sampler_list = [LeapHybridSampler()]
    # sampler_list = [LeapHybridSampler(), neal.SimulatedAnnealingSampler(), greedy.SteepestDescentSampler(),
                    # tabu.TabuSampler()]
    results_folder_path = f'{os.path.abspath(args.ouput)}/'
    # clean_results(results_folder_path, data_reader_classes, method_list, sampler_list, recommender_list, args.kfolds)
    main(data_reader_classes, method_list, sampler_list, recommender_list, results_folder_path,
         args.kfolds, args.T, args.alpha, args.beta, args.implicit)
