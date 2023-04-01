import os
import time
import copy
import shutil

import argparse
import dimod
import greedy
import neal
import numpy as np
import tabu
from dwave.system import LeapHybridSampler

from CommunityDetection import BaseCommunityDetection, QUBOBipartiteCommunityDetection, \
    QUBOBipartiteProjectedCommunityDetection, Communities, CommunityDetectionRecommender, \
    get_community_folder_path, KmeansCommunityDetection, HierarchicalClustering, \
    QUBOGraphCommunityDetection, QUBOProjectedCommunityDetection, UserCommunityDetection, \
    HybridCommunityDetection, MultiHybridCommunityDetection, QUBONcutCommunityDetection, \
    SpectralClustering
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, MovielensSampleReader
from recsys.Evaluation.Evaluator import EvaluatorHoldout
from recsys.Recommenders.BaseRecommender import BaseRecommender
from recsys.Recommenders.NonPersonalizedRecommender import TopPop
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.urm import get_community_urm, load_data, merge_sparse_matrices
from utils.plot import plot_metric, plot_cut, plot_divide
from results.read_results import print_result

CUTOFF_LIST = [5, 10, 20, 30, 40, 50, 100]
ADAPATIVE_METRIC = ['PRECISION', 'MAP', 'NDCG'][2] 
ADAPATIVE_DATA = ['validation', 'test'][0]


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


def train_all_data_recommender(recommender: Type[BaseRecommender], urm_train_last_test, urm_test, dataset_name: str,
                               results_folder_path: str):
    recommender_name = recommender.RECOMMENDER_NAME
    output_folder_path = f'{results_folder_path}{dataset_name}/{recommender_name}/'

    evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=CUTOFF_LIST)

    print(f'Training {recommender_name} on all data...')

    time_on_train = time.time()
    rec = recommender(urm_train_last_test)
    rec.fit()
    time_on_train = time.time() - time_on_train

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


def get_recommender_on_community(recommender, community, urm_train):
    c_urm_train, _, _ = get_community_urm(urm_train, community=community, filter_items=False)
    comm_recommender = recommender(c_urm_train)
    comm_recommender.fit()
    return comm_recommender


def train_recommender_on_community(recommender, community, urm_train, urm_validation, urm_test, dataset_name,
                                   results_folder_path, method_folder_path, n_iter=0, n_comm=None, folder_suffix='',
                                   **kwargs):
    """
    validation: fit(train), predict(validation)
    test: fit(train + validation), predict(test)
    baseline: best_model fit(train + validation), predict(test)
    """
    recommender_name = recommender.RECOMMENDER_NAME
    print(f'Training {recommender_name} on community {n_comm if n_comm is not None else ""} of iteration {n_iter}...')
    print(f"community.shape({len(community.users)}, {len(community.items)})")

    output_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                   folder_suffix=folder_suffix)
    output_folder_path = f'{output_folder_path}{recommender_name}/'

    base_recommender_path = f'{results_folder_path}{dataset_name}/{recommender_name}/'

    c_urm_train, _, _ = get_community_urm(urm_train, community=community, filter_items=False)
    c_urm_validation, _, _ = get_community_urm(urm_validation, community=community, filter_items=False)
    c_urm_test, _, _ = get_community_urm(urm_test, community=community, filter_items=False)
    c_urm_train_last_test = merge_sparse_matrices(c_urm_train, c_urm_validation)

    ignore_users = np.arange(c_urm_train_last_test.shape[0])[np.logical_not(community.user_mask)]
    evaluator_validation = EvaluatorHoldout(c_urm_validation, cutoff_list=CUTOFF_LIST, ignore_users=ignore_users)
    evaluator_test = EvaluatorHoldout(c_urm_test, cutoff_list=CUTOFF_LIST, ignore_users=ignore_users)

    time_on_train = time.time()
    validation_recommender = recommender(c_urm_train)
    validation_recommender.fit()
    time_on_train = time.time() - time_on_train

    time_on_validation = time.time()
    result_df, result_string = evaluator_validation.evaluateRecommender(validation_recommender)
    time_on_validation = time.time() - time_on_validation

    data_dict_to_save = {
        'result_df': result_df,
        'result_string': result_string,
        'time_on_train': time_on_train,
        'time_on_validation': time_on_validation,
    }

    output_dataIO = DataIO(output_folder_path)
    output_dataIO.save_data('validation', data_dict_to_save)
    community.result_dict_validation = copy.deepcopy(data_dict_to_save)

    time_on_train = time.time()
    comm_recommender = recommender(c_urm_train_last_test)
    comm_recommender.fit()
    time_on_train = time.time() - time_on_train

    time_on_test = time.time()
    result_df, result_string = evaluator_test.evaluateRecommender(comm_recommender)
    time_on_test = time.time() - time_on_test

    data_dict_to_save = {
        'result_df': result_df,
        'result_string': result_string,
        'time_on_train': time_on_train,
        'time_on_test': time_on_test,
    }

    output_dataIO = DataIO(output_folder_path)
    output_dataIO.save_data('test', data_dict_to_save)
    community.result_dict_test = copy.deepcopy(data_dict_to_save)

    print(f'Evaluating base model on community {n_comm if n_comm is not None else ""} of iteration {n_iter}...')
    base_recommender = recommender(c_urm_train_last_test)
    recommender_file_name = f'{recommender_name}_best_model_last'
    base_recommender.load_model(base_recommender_path, recommender_file_name)
    base_evaluator_test = EvaluatorHoldout(c_urm_test, cutoff_list=CUTOFF_LIST, ignore_users=ignore_users)

    time_on_test = time.time()
    result_df, result_string = base_evaluator_test.evaluateRecommender(base_recommender)
    time_on_test = time.time() - time_on_test

    baseline_dict = {
        'result_df': result_df,
        'result_string': result_string,
        'time_on_test': time_on_test,
    }
    output_dataIO.save_data('baseline', baseline_dict)
    community.result_dict_baseline = copy.deepcopy(baseline_dict)
    print(result_string)

    return comm_recommender


def evaluate_recommender(urm_train_last_test, urm_test, communities, recommenders, output_folder_path=None, recommender_name=None,
                         n_iter=None):
    print(f'Evaluating {recommender_name} on the result of community detection.')

    recommender = CommunityDetectionRecommender(urm_train_last_test, communities=communities, recommenders=recommenders,
                                                n_iter=n_iter)

    evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=CUTOFF_LIST)
    time_on_test = time.time()
    result_df, result_string = evaluator_test.evaluateRecommender(recommender)
    time_on_test = time.time() - time_on_test

    result_dict = {
        'result_df': result_df,
        'result_string': result_string,
        'time_on_test': time_on_test,
    }
    
    if output_folder_path is not None:
        dataIO = DataIO(output_folder_path)
        dataIO.save_data(f'cd_{recommender_name}', result_dict)

    print(result_string)
    return result_dict


def main(data_reader_classes, method_list: Iterable[Type[BaseCommunityDetection]],
         sampler_list: Iterable[dimod.Sampler], recommender_list: Iterable[Type[BaseRecommender]],
         result_folder_path: str):
    split_quota = [80, 10, 10]
    user_wise = False
    make_implicit = False
    threshold = None

    recsys_args = {
        'cutoff_to_optimize': 10,
        'cutoff_list': CUTOFF_LIST,
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
        urm_train, urm_validation, urm_test = load_data(data_reader, split_quota=split_quota, user_wise=user_wise,
                                                        make_implicit=make_implicit, threshold=threshold)

        urm_train_last_test = merge_sparse_matrices(urm_train, urm_validation)

        for recommender in recommender_list:
            recommender_name = recommender.RECOMMENDER_NAME
            output_folder_path = f'{result_folder_path}{dataset_name}/{recommender_name}/'
            if not os.path.exists(f'{output_folder_path}baseline.zip') or not os.path.exists(
                    f'{output_folder_path}{recommender_name}_best_model_last.zip'):
                train_all_data_recommender(recommender, urm_train_last_test, urm_test, dataset_name, result_folder_path)
            else:
                print(f'{recommender_name} already trained and evaluated on {dataset_name}.')

        for method in method_list:
            recommend_per_method(urm_train, urm_validation, urm_test, urm_train_last_test, method, sampler_list,
                                 recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy,
                                 save_model=save_model)


def recommend_per_method(urm_train, urm_validation, urm_test, cd_urm, method, sampler_list, recommender_list,
                         dataset_name, folder_path, **kwargs):
    if method.is_qubo:
        for sampler in sampler_list:
            cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, method, recommender_list, dataset_name,
                              folder_path, sampler=sampler, **kwargs)
    else:
        cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, method, recommender_list, dataset_name,
                          folder_path, **kwargs)


def adaptive_selection(num_iters, urm_train, urm_test, recommender, output_folder_path, communities: Communities = None):
    if communities.s0 is not None:
        adaptive_selection(num_iters, urm_train, urm_test, recommender, output_folder_path, communities.s0)
    if communities.s1 is not None:
        adaptive_selection(num_iters, urm_train, urm_test, recommender, output_folder_path, communities.s1)

    def get_result_dict():
        cd_recommenders = []
        for community in communities.iter():
            comm_recommender = get_recommender_on_community(recommender, community, urm_train)
            cd_recommenders.append(comm_recommender)
        return evaluate_recommender(urm_train, urm_test, communities, cd_recommenders)
    
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
        CutOff = 10
        Metrics = ADAPATIVE_METRIC
        result_df_0 = result_dict_0['result_df']
        result_df_1 = result_dict_1['result_df']
        metric_0 = result_df_0.loc[CutOff, Metrics]
        metric_1 = result_df_1.loc[CutOff, Metrics]
        if metric_1 > 0.0:
            return (metric_0 - metric_1) / metric_1
        else:
            return 0.0
        # return metric_0 > metric_1

    print("-----------------------------")
    print(f"Communities: num_iter{communities.num_iters}, user: {communities.n_users}, items: {communities.n_items}")
    ratio = compare_result(result_dict_divide, result_dict_combine)
    communities.divide_info = ratio
    if ADAPATIVE_DATA == 'test':
        threshold = 0
    elif ADAPATIVE_DATA == 'validation':
        threshold = (num_iters - communities.num_iters) * 0.025
    if ratio > threshold:
        communities.divide_flag = True
        communities.result_dict_test = result_dict_divide
        print('choose divide')
    else:
        communities.divide_flag = False
        communities.result_dict_test = result_dict_combine
        print('choose combine')
    print("-----------------------------")


def cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, method, recommender_list, dataset_name, folder_path,
                      sampler: dimod.Sampler = None, **kwargs):
    dataset_folder_path = f'{folder_path}{dataset_name}/'
    communities = load_communities(dataset_folder_path, method, sampler)
    if communities is None:
        print(f'Could not load communitites for {dataset_folder_path}, {method}, {sampler}.')
        return

    method_folder_path = f'{folder_path}{dataset_name}/{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}/'
    output_folder_path = get_community_folder_path(method_folder_path, n_iter=-1, folder_suffix=folder_suffix)

    num_iters = communities.num_iters + 1
    for n_iter in range(num_iters):
        recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, method, recommender_list, dataset_name,
                           folder_path, sampler=sampler, communities=communities, n_iter=n_iter, **kwargs)
    plot_metric(communities, method_folder_path)

    for recommender in recommender_list:
        adaptive_communities = copy.deepcopy(communities)
        if ADAPATIVE_DATA == 'validation':
            adaptive_selection(num_iters - 1, urm_train, urm_validation, recommender, output_folder_path, adaptive_communities)
        elif ADAPATIVE_DATA == 'test':
            adaptive_selection(num_iters - 1, cd_urm, urm_test, recommender, output_folder_path, adaptive_communities)
        cd_recommenders = []
        n_comm = 0
        for community in adaptive_communities.iter():
            comm_recommender = train_recommender_on_community(recommender, community, urm_train, urm_validation,
                                                              urm_test, dataset_name, folder_path,
                                                              method_folder_path, n_iter=-1, n_comm=n_comm,
                                                              folder_suffix=folder_suffix)
            cd_recommenders.append(comm_recommender)
            n_comm += 1
        evaluate_recommender(cd_urm, urm_test, adaptive_communities, cd_recommenders, output_folder_path,
                             recommender.RECOMMENDER_NAME)
        plot_metric(adaptive_communities, output_folder_path, 10, ADAPATIVE_METRIC)
        plot_divide(adaptive_communities, output_folder_path)


def recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, method, recommender_list, dataset_name, folder_path,
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
            for community in communities.iter(n_iter):
                comm_recommender = train_recommender_on_community(recommender, community, urm_train, urm_validation,
                                                                  urm_test, dataset_name, folder_path,
                                                                  method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                                  folder_suffix=folder_suffix)
                cd_recommenders.append(comm_recommender)
                n_comm += 1
            evaluate_recommender(cd_urm, urm_test, communities, cd_recommenders, output_folder_path,
                                 recommender_name, n_iter=n_iter)
        else:
            print('Recommender already trained and evaluated.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.5)
    parser.add_argument('-b', '--beta', type=float, default=0.25)
    args = parser.parse_args()
    return args

def clean_results(result_folder_path, data_reader_classes, method_list, sampler_list):
    for data_reader_class in data_reader_classes:
        data_reader = data_reader_class()
        dataset_name = data_reader._get_dataset_name()
        dataset_folder_path = f'{result_folder_path}{dataset_name}/'
        if not os.path.exists(dataset_folder_path):
            continue
        hybrid_folder_path = os.path.join(dataset_folder_path, 'Hybrid')
        if os.path.exists(hybrid_folder_path):
            shutil.rmtree(hybrid_folder_path)
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
                    result_file = os.path.join(sampler_folder_path, 'cd_TopPopRecommender.zip')
                    if os.path.exists(result_file):
                        # print('remove: ', result_file)
                        os.remove(result_file)
                    for c in os.listdir(sampler_folder_path):
                        c_folder_path = os.path.join(sampler_folder_path, c)
                        if os.path.isdir(c_folder_path) and c[0] == 'c':
                            # print('remove: ', c_folder_path)
                            shutil.rmtree(c_folder_path)


def save_results(data_reader_classes, result_folder_path, *args):
    tag = []
    for arg in args:
        if arg is None:
            continue
        tag.append(str(arg))
    tag = '_'.join(tag) if tag else '_'

    for data_reader in data_reader_classes:
        dataset_name = data_reader.DATASET_SUBFOLDER
        output_folder = os.path.join(result_folder_path, dataset_name, 'results')
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_folder = os.path.join(output_folder, tag) 
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        print_result(dataset_name, False, output_folder)


if __name__ == '__main__':
    args = parse_args()
    data_reader_classes = [MovielensSampleReader]
    # data_reader_classes = [Movielens1MReader]
    # data_reader_classes = [Movielens100KReader, Movielens1MReader, FilmTrustReader, MovielensHetrec2011Reader,
    #                        LastFMHetrec2011Reader, FrappeReader, CiteULike_aReader, CiteULike_tReader]
    recommender_list = [TopPop]
    # method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection, UserCommunityDetection]
    # method_list = [SpectralClustering]
    method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection]
    # sampler_list = [neal.SimulatedAnnealingSampler()]
    sampler_list = [greedy.SteepestDescentSampler(), tabu.TabuSampler()]
    # sampler_list = [LeapHybridSampler(), neal.SimulatedAnnealingSampler(), greedy.SteepestDescentSampler(),
                    # tabu.TabuSampler()]
    result_folder_path = './results/'
    clean_results(result_folder_path, data_reader_classes, method_list, sampler_list)
    main(data_reader_classes, method_list, sampler_list, recommender_list, result_folder_path)
    # save_results(data_reader_classes, result_folder_path, args.alpha, args.beta)
    save_results(data_reader_classes, result_folder_path, args.alpha)
