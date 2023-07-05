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

from CommunityDetection import BaseCommunityDetection, QUBOBipartiteCommunityDetection, \
    QUBOBipartiteProjectedCommunityDetection, Communities, CommunityDetectionRecommender, \
    get_community_folder_path, KmeansCommunityDetection, HierarchicalClustering, \
    QUBOGraphCommunityDetection, QUBOProjectedCommunityDetection, UserCommunityDetection, \
    HybridCommunityDetection, MultiHybridCommunityDetection, QUBONcutCommunityDetection, \
    SpectralClustering, QUBOBipartiteProjectedItemCommunityDetection, CommunitiesEI, \
    LTBipartiteProjectedCommunityDetection, LTBipartiteCommunityDetection, QuantityDivision, \
    METHOD_DICT, get_cascade_class, UserBipartiteCommunityDetection
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, \
    MovielensSampleReader, MovielensSample2Reader
# from recsys.Evaluation.Evaluator import EvaluatorHoldout
from recsys.Evaluation.EvaluatorSeparate import EvaluatorSeparate
from recsys.Recommenders.BaseRecommender import BaseRecommender
from recsys.Recommenders import SVRRecommender, LRRecommender, DTRecommender, RECOMMENDER_DICT
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.urm import get_community_urm, load_data, merge_sparse_matrices, head_tail_cut
from utils.plot import plot_line, plot_scatter, plot_divide, plot_metric
from utils.derived_variables import create_related_variables
# from results.read_results import print_result
from results.plot_results import print_result

logging.basicConfig(level=logging.INFO)
CUT_RATIO: float = None
EI: bool = False # EI if True else (TC or CT)
ADAPATIVE_FLAG = False
ADAPATIVE_METRIC = ['MAE', 'MSE', 'W-MAE', 'W-RMSE'][1] 
ADAPATIVE_DATA = ['validation', 'test'][1]
MIN_RATINGS_PER_USER = 1
EVALUATE_FLAG = True


def load_communities(folder_path, method, sampler=None, n_iter=0, n_comm=None):
    method_folder_path = f'{folder_path}{method.name}/'
    # method_folder_path = os.path.join(folder_path, method.name)
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
                               results_folder_path: str):
    recommender_name = recommender.RECOMMENDER_NAME
    output_folder_path = f'{results_folder_path}{dataset_name}/{recommender_name}/'
    # output_folder_path = os.path.join(results_folder_path, dataset_name, recommender_name)

    evaluator_test = EvaluatorSeparate(urm_test)

    print(f'Training {recommender_name} on all data...')

    time_on_train = time.time()
    rec = recommender(urm_train_last_test, ucm, icm)
    rec.fit()
    time_on_train = time.time() - time_on_train

    if not EVALUATE_FLAG:
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


def train_recommender_on_community(recommender, community, urm_train, urm_validation, urm_test, ucm, icm, dataset_name,
                                   results_folder_path, method_folder_path, n_iter=0, n_comm=None, folder_suffix='',
                                   **kwargs):
    """
    validation: fit(train), predict(validation)
    test: fit(train + validation), predict(test)
    baseline: best_model fit(train + validation), predict(test)
    """
    recommender_name = recommender.RECOMMENDER_NAME
    print(f'Training {recommender_name} on community {n_comm if n_comm is not None else ""} of iteration {n_iter}...')
    logging.info(f"community.size({len(community.users)}, {len(community.items)})")
    logging.info(f"community.shape({len(community.user_mask)}, {len(community.item_mask)})")
    logging.debug(f'urm.shape: {urm_train.shape}')

    output_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                   folder_suffix=folder_suffix)
    output_folder_path = f'{output_folder_path}{recommender_name}/'
    # output_folder_path = os.path.join(output_folder_path, recommender_name)

    base_recommender_path = f'{results_folder_path}{dataset_name}/{recommender_name}/'
    # base_recommender_path = os.path.join(results_folder_path, dataset_name, recommender_name)

    c_urm_train, _, _, c_icm, c_ucm = get_community_urm(urm_train, community=community, filter_items=False, icm=icm, ucm=ucm)
    c_urm_validation, _, _ = get_community_urm(urm_validation, community=community, filter_items=False)
    c_urm_test, _, _ = get_community_urm(urm_test, community=community, filter_items=False)
    c_urm_train_last_test = merge_sparse_matrices(c_urm_train, c_urm_validation)

    ignore_users = np.arange(c_urm_train_last_test.shape[0])[np.logical_not(community.user_mask)]
    evaluator_validation = EvaluatorSeparate(c_urm_validation, ignore_users=ignore_users)
    evaluator_test = EvaluatorSeparate(c_urm_test, ignore_users=ignore_users)

    if EVALUATE_FLAG:
        time_on_train = time.time()
        validation_recommender = recommender(c_urm_train, c_ucm, c_icm, community.users)
        validation_recommender.fit()
        time_on_train = time.time() - time_on_train

        time_on_validation = time.time()
        result_df, result_string = evaluator_validation.evaluateRecommender(validation_recommender)
        time_on_validation = time.time() - time_on_validation

        data_dict_to_save = {
            'result_df': result_df,
            'time_on_train': time_on_train,
            'time_on_validation': time_on_validation,
        }

        output_dataIO = DataIO(output_folder_path)
        output_dataIO.save_data('validation', data_dict_to_save)
        community.result_dict_validation = copy.deepcopy(data_dict_to_save)

    time_on_train = time.time()
    comm_recommender = recommender(c_urm_train_last_test, c_ucm, c_icm, community.users)
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
        base_recommender = recommender(c_urm_train_last_test, c_ucm, c_icm)
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
         result_folder_path: str):
    global EI, CUT_RATIO
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

        h_urm_train, h_urm_validation, h_urm_test, h_icm, h_ucm,\
        t_urm_train, t_urm_validation, t_urm_test, t_icm, t_ucm = \
            head_tail_cut(CUT_RATIO, urm_train, urm_validation, urm_test, icm, ucm)
        head_flag = h_urm_train.shape[0] > 0
        logging.info(f'head shape: {h_urm_train.shape}, tail shape: {t_urm_train.shape}')

        t_urm_train_last_test = merge_sparse_matrices(t_urm_train, t_urm_validation)
        if head_flag:
            h_urm_train_last_test = merge_sparse_matrices(h_urm_train, h_urm_validation)

        for recommender in recommender_list:
            recommender_name = recommender.RECOMMENDER_NAME
            output_folder_path = f'{result_folder_path}{dataset_name}/{recommender_name}/'
            # output_folder_path = os.path.join(result_folder_path, dataset_name, recommender_name)
            if not os.path.exists(f'{output_folder_path}baseline.zip') or not os.path.exists(
                    f'{output_folder_path}{recommender_name}_best_model_last.zip'):
                train_all_data_recommender(recommender, t_urm_train_last_test, t_urm_test, t_ucm, t_icm, dataset_name, result_folder_path)
            else:
                print(f'{recommender_name} already trained and evaluated on {dataset_name}.')
        for method in method_list:
            logging.info(f'------------start {method.name}----------')
            recommend_per_method(t_urm_train, t_urm_validation, t_urm_test, t_urm_train_last_test, t_ucm, t_icm, method, sampler_list,
                                 recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy,
                                 save_model=save_model, each_item=EI)
            if not head_flag or EI:
                continue
            recommend_per_method(h_urm_train, h_urm_validation, h_urm_test, h_urm_train_last_test, h_ucm, h_icm, method, sampler_list,
                                 recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy,
                                 save_model=save_model, each_item=True)


def recommend_per_method(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, sampler_list, recommender_list,
                         dataset_name, folder_path, **kwargs):
    if method.is_qubo:
        for sampler in sampler_list:
            cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name,
                              folder_path, sampler=sampler, **kwargs)
    else:
        cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name,
                          folder_path, **kwargs)


def adaptive_selection(num_iters, urm_train, urm_test, ucm, icm, recommender, output_folder_path, communities: Communities = None):
    if communities.s0 is not None:
        adaptive_selection(num_iters, urm_train, urm_test, ucm, icm, recommender, output_folder_path, communities.s0)
    if communities.s1 is not None:
        adaptive_selection(num_iters, urm_train, urm_test, ucm, icm, recommender, output_folder_path, communities.s1)

    def get_result_dict():
        cd_recommenders = []
        n_users, n_items = urm_train.shape
        user_mask = np.zeros(n_users).astype(bool)
        n_comm = 0
        for community in communities.iter():
            user_mask[community.user_mask] = True
            c_urm_train, _, _, c_icm, c_ucm = get_community_urm(urm_train, community=community, filter_items=False, icm=icm, ucm=ucm)
            comm_recommender = recommender(c_urm_train, c_ucm, c_icm, community.users)
            comm_recommender.fit()
            cd_recommenders.append(comm_recommender)
            n_comm += 1
        logging.info(f'divide to {n_comm} communities.')
        ignore_users = np.arange(n_users)[np.logical_not(user_mask)]
        # return evaluate_recommender(urm_train, urm_test, ucm, icm, communities, cd_recommenders,
        return evaluate_recommender(urm_train, urm_train, ucm, icm, communities, cd_recommenders,
                                    ignore_users=ignore_users, min_ratings_per_user=MIN_RATINGS_PER_USER)
    
    result_dict_divide = get_result_dict()
    if communities.s0 is None and communities.s1 is None:
        result_dict_combine = result_dict_divide
    else:
        communities.divide_flag = False
        result_dict_combine = get_result_dict()

    def compare_result(result_dict_0: dict, result_dict_1: dict) -> float:
        '''
        return (result_dict_0 - result_dict_1) / result_dict_1
        '''
        def accumulate_result(result_df):
            sum = 0.0
            tot = 0
            for i in range(len(result_df)):
                metric = result_df.loc[i, ADAPATIVE_METRIC]
                # cnt = result_df.loc[i, 'num_rating']
                # sum += metric * cnt
                # tot += cnt
                sum += metric
                tot += 1
            return sum / tot if tot > 0 else 0.0

        result_df_0 = result_dict_0['result_df']
        result_df_1 = result_dict_1['result_df']
        metric_0 = accumulate_result(result_df_0)
        metric_1 = accumulate_result(result_df_1)
        logging.info(f'divide compare combine, {metric_0} : {metric_1}')
        if metric_1 > 0.0:
            return (metric_0 - metric_1) / metric_1
        else:
            return 0.0
        # return metric_0 > metric_1

    num_users = sum(communities.user_mask)
    num_items = len(communities.item_mask)
    logging.info(f"Communities: num_iter {communities.num_iters}, user: {num_users}, items: {num_items}")
    ratio = compare_result(result_dict_divide, result_dict_combine)
    communities.divide_info = ratio
    if ADAPATIVE_DATA == 'test':
        threshold = 0
    elif ADAPATIVE_DATA == 'validation':
        # threshold = (num_iters - communities.num_iters) * 0.01
        threshold = 0
    if ratio < threshold:
        communities.divide_flag = True
        communities.result_dict_test = result_dict_divide
        logging.info('choose divide')
    else:
        communities.divide_flag = False
        communities.result_dict_test = result_dict_combine
        logging.info('choose combine')


def cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name, folder_path,
                      sampler: dimod.Sampler = None, each_item: bool = False, **kwargs):
    dataset_folder_path = f'{folder_path}{dataset_name}/'
    communities = load_communities(dataset_folder_path, method, sampler)
    if communities is None:
        print(f'Could not load communitites for {dataset_folder_path}, {method}, {sampler}.')
        return

    if each_item:
        n_users, n_items = urm_train.shape
        recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name,
                           folder_path, sampler=sampler, communities=CommunitiesEI(n_users, n_items), n_iter=-1, **kwargs)
    else:
        num_iters = communities.num_iters + 1
        for n_iter in range(num_iters):
            recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name,
                               folder_path, sampler=sampler, communities=communities, n_iter=n_iter, **kwargs)
    # plot_metric(communities, method_folder_path)
    
    method_folder_path = f'{folder_path}{dataset_name}/{method.name}/'
    # method_folder_path = os.path.join(folder_path, dataset_name, method.name)
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}/'
    output_folder_path = get_community_folder_path(method_folder_path, n_iter=-1, folder_suffix=folder_suffix)
    if not ADAPATIVE_FLAG:
        return
    logging.info('start adaptive selection')
    for recommender in recommender_list:
        adaptive_communities = copy.deepcopy(communities)
        if ADAPATIVE_DATA == 'validation':
            adaptive_selection(num_iters - 1, urm_train, urm_validation, ucm, icm, recommender, output_folder_path, adaptive_communities)
        elif ADAPATIVE_DATA == 'test':
            adaptive_selection(num_iters - 1, cd_urm, urm_test, ucm, icm, recommender, output_folder_path, adaptive_communities)
        cd_recommenders = []
        n_comm = 0
        for community in adaptive_communities.iter():
            comm_recommender = train_recommender_on_community(recommender, community, urm_train, urm_validation,
                                                              urm_test, ucm, icm, dataset_name, folder_path,
                                                              method_folder_path, n_iter=-1, n_comm=n_comm,
                                                              folder_suffix=folder_suffix)
            cd_recommenders.append(comm_recommender)
            n_comm += 1
        evaluate_recommender(cd_urm, urm_test, ucm, icm, adaptive_communities, cd_recommenders, output_folder_path,
                            #  recommender.RECOMMENDER_NAME, min_ratings_per_user=MIN_RATINGS_PER_USER)
                             recommender.RECOMMENDER_NAME)
        logging.info(f'adaptive_communities has {n_comm} communities.')
        # plot_metric(adaptive_communities, output_folder_path, 10, ADAPATIVE_METRIC)
        plot_divide(adaptive_communities, output_folder_path)


def recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name, folder_path,
                       sampler: dimod.Sampler = None, communities: Communities = None, n_iter: int = 0, **kwargs):
    method_folder_path = f'{folder_path}{dataset_name}/{method.name}/'
    # method_folder_path = os.path.join(folder_path, dataset_name, method.name)
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
    parser.add_argument('-r', '--recommender', nargs='+', type=str, default='LRRecommender', help='recommender',
                        choices=['LRRecommender', 'SVRRecommender', 'DTRecommender'])
    parser.add_argument('-c', '--cut_ratio', type=float, default=0.0, help='head ratio for clustered tail')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha for cascade')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='beta for quantity')
    parser.add_argument('-t', '--T', type=int, default=5, help='T for quantity')
    parser.add_argument('-l', '--layer', type=int, default=0, help='number of layer of quantity')
    parser.add_argument('-o', '--ouput', type=str, default='results', help='the path to save the result')
    parser.add_argument('--attribute', action='store_true', help='Use item attribute data (cascade) or not')
    args = parser.parse_args()
    return args


def clean_results(result_folder_path, data_reader_classes, method_list, sampler_list, recommender_list):
    for data_reader_class in data_reader_classes:
        data_reader = data_reader_class()
        dataset_name = data_reader._get_dataset_name()
        dataset_folder_path = f'{result_folder_path}{dataset_name}/'
        # dataset_folder_path = os.path.join(result_folder_path, dataset_name)
        if not os.path.exists(dataset_folder_path):
            continue
        hybrid_folder_path = os.path.join(dataset_folder_path, 'Hybrid')
        logging.debug(f'clean {hybrid_folder_path}')
        if os.path.exists(hybrid_folder_path):
            shutil.rmtree(hybrid_folder_path)
        # for recommender in recommender_list:
        #     recommender_folder_path = os.path.join(dataset_folder_path, recommender.RECOMMENDER_NAME)
        #     logging.debug(f'clean {recommender_folder_path}')
        #     if os.path.exists(recommender_folder_path):
        #         shutil.rmtree(recommender_folder_path)
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
                for sample in sampler_list:
                    # sampler_folder_path = os.path.join(iter_folder_path, sample.__name__)
                    sampler_folder_path = os.path.join(iter_folder_path, 'SimulatedAnnealingSampler')
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


def save_results(data_reader_classes, result_folder_path, method_list, sampler_list, recommender_list, *args):
    global CUT_RATIO
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
    CUT_RATIO = args.cut_ratio
    data_reader_classes = [Movielens100KReader]
    # data_reader_classes = [Movielens1MReader]
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
    result_folder_path = f'{os.path.abspath(args.ouput)}/'
    # clean_results(result_folder_path, data_reader_classes, method_list, sampler_list, recommender_list)
    main(data_reader_classes, method_list, sampler_list, recommender_list, result_folder_path)
    # save_results(data_reader_classes, result_folder_path, method_list, args.T, args.alpha, args.cut_ratio)
    save_results(data_reader_classes, result_folder_path, method_list, sampler_list, recommender_list, args.T, args.alpha, args.beta, args.cut_ratio)
