import os
import time
import copy
import shutil
from typing import List

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
    SpectralClustering, QUBOBipartiteProjectedItemCommunityDetection
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, \
    MovielensSampleReader, MovielensSample2Reader
# from recsys.Evaluation.Evaluator import EvaluatorHoldout
from recsys.Evaluation.AC_Evaluator import AC_Evaluator
from recsys.Recommenders.BaseRecommender import BaseRecommender
# from recsys.Recommenders.NonPersonalizedRecommender import TopPop
from recsys.Recommenders.AdaptiveClustering import AdaptiveClustering
from recsys.Recommenders.CommunityDetectionAdaptiveClustering import CommunityDetectionAdaptiveClustering
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.urm import get_community_urm, load_data, merge_sparse_matrices
from utils.plot import plot_lines
from results.read_results import print_result

CRITERION: int = None
MAE_data = {}
RMSE_data = {}

def plot(urm, method, dataset_name, folder_path):
    method_folder_path = f'{folder_path}{dataset_name}/{method.name}/'
    I_quantity = np.ediff1d(urm.tocsc().indptr) # count of each colum
    n_items = I_quantity.size

    print(MAE_data)
    print(RMSE_data)
    for data in (MAE_data, RMSE_data):
        for key in data: # sort data according I_quantity
            data[key] = [x for _, x in sorted(zip(I_quantity, data[key]))]
    I_quantity = sorted(I_quantity)
    x = range(n_items)
    plot_lines(x, MAE_data, method_folder_path, 'item rank', 'MAE')
    plot_lines(x, RMSE_data, method_folder_path, 'item rank', 'RMSE')

    print(MAE_data)
    print(RMSE_data)
    x = list(set(I_quantity))
    for data in (MAE_data, RMSE_data):
        for key in data:
            new_data = {}
            cnt = {}
            for i in range(n_items):
                quantity = I_quantity[i]
                new_data[quantity] = new_data.get(quantity, 0) + data[key][i]
                cnt[quantity] = cnt.get(quantity, 0) + 1
            for k in new_data:
                new_data[k] /= cnt[k]
            data[key] = list(new_data.values())
            data[key]
    plot_lines(x, MAE_data, method_folder_path, 'the number of ratings', 'MAE')
    plot_lines(x, RMSE_data, method_folder_path, 'the number of ratings', 'RMSE')


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
                               results_folder_path: str):
    """
    get result of EI and AC
    """
    recommender_name = recommender.RECOMMENDER_NAME
    output_folder_path = f'{results_folder_path}{dataset_name}/{recommender_name}/'

    evaluator_test = AC_Evaluator(urm_test)

    print(f'Training {recommender_name} on all data...')

    # EI
    time_on_train = time.time()
    rec = recommender(urm_train_last_test, ucm, icm)
    rec.fit()
    time_on_train = time.time() - time_on_train

    time_on_test = time.time()
    MAE, RMSE, MAE_all, RMSE_all = evaluator_test.evaluateRecommender(rec)
    time_on_test = time.time() - time_on_test

    MAE_data['EI'] = MAE
    RMSE_data['EI'] = RMSE
    data_dict_to_save = {
        'MAE': MAE,
        'RMSE': RMSE,
        'time_on_train': time_on_train,
        'time_on_test': time_on_test,
    }

    output_dataIO = DataIO(output_folder_path)
    output_dataIO.save_data('EI', data_dict_to_save)

    rec.save_model(output_folder_path, f'{recommender_name}_EI')

    # AC
    time_on_train = time.time()
    rec = recommender(urm_train_last_test, ucm, icm, CRITERION)
    rec.fit()
    time_on_train = time.time() - time_on_train

    time_on_test = time.time()
    MAE, RMSE, MAE_all, RMSE_all = evaluator_test.evaluateRecommender(rec)
    time_on_test = time.time() - time_on_test

    MAE_data['AC'] = MAE
    RMSE_data['AC'] = RMSE
    data_dict_to_save = {
        'MAE': MAE,
        'RMSE': RMSE,
        'time_on_train': time_on_train,
        'time_on_test': time_on_test,
    }

    output_dataIO = DataIO(output_folder_path)
    output_dataIO.save_data('AC', data_dict_to_save)

    rec.save_model(output_folder_path, f'{recommender_name}_AC')


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
    print(f"community.shape({len(community.users)}, {len(community.items)})")

    output_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                   folder_suffix=folder_suffix)
    output_folder_path = f'{output_folder_path}{recommender_name}/'

    base_recommender_path = f'{results_folder_path}{dataset_name}/{recommender_name}/'

    c_urm_train, _, _, c_icm, c_ucm = get_community_urm(urm_train, community=community, filter_items=False, icm=icm, ucm=ucm)
    c_urm_validation, _, _ = get_community_urm(urm_validation, community=community, filter_items=False)
    c_urm_test, _, _ = get_community_urm(urm_test, community=community, filter_items=False)
    c_urm_train_last_test = merge_sparse_matrices(c_urm_train, c_urm_validation)

    ignore_users = np.arange(c_urm_train_last_test.shape[0])[np.logical_not(community.user_mask)]
    evaluator_validation = AC_Evaluator(c_urm_validation, ignore_users=ignore_users)
    evaluator_test = AC_Evaluator(c_urm_test, ignore_users=ignore_users)

    time_on_train = time.time()
    validation_recommender = recommender(c_urm_train, c_ucm, c_icm, CRITERION, community)
    validation_recommender.fit()
    time_on_train = time.time() - time_on_train

    time_on_validation = time.time()
    MAE, RMSE, MAE_all, RMSE_all = evaluator_validation.evaluateRecommender(validation_recommender)
    time_on_validation = time.time() - time_on_validation

    data_dict_to_save = {
        'MAE': MAE,
        'RMSE': RMSE,
        'time_on_train': time_on_train,
        'time_on_test': time_on_test,
    }

    output_dataIO = DataIO(output_folder_path)
    output_dataIO.save_data('validation', data_dict_to_save)
    community.result_dict_validation = copy.deepcopy(data_dict_to_save)

    time_on_train = time.time()
    comm_recommender = recommender(c_urm_train_last_test, c_ucm, c_icm, CRITERION, community)
    comm_recommender.fit()
    time_on_train = time.time() - time_on_train

    time_on_test = time.time()
    MAE, RMSE, MAE_all, RMSE_all = evaluator_test.evaluateRecommender(comm_recommender)
    time_on_test = time.time() - time_on_test

    data_dict_to_save = {
        'MAE': MAE,
        'RMSE': RMSE,
        'time_on_train': time_on_train,
        'time_on_test': time_on_test,
    }

    output_dataIO = DataIO(output_folder_path)
    output_dataIO.save_data('test', data_dict_to_save)
    community.result_dict_test = copy.deepcopy(data_dict_to_save)

    print(f'Evaluating base model on community {n_comm if n_comm is not None else ""} of iteration {n_iter}...')
    base_recommender = recommender(c_urm_train_last_test, c_ucm, c_icm, CRITERION, community)
    recommender_file_name = f'{recommender_name}_best_model_last'
    base_recommender.load_model(base_recommender_path, recommender_file_name)
    base_evaluator_test = AC_Evaluator(c_urm_test, ignore_users=ignore_users)

    time_on_test = time.time()
    MAE, RMSE, MAE_all, RMSE_all = base_evaluator_test.evaluateRecommender(base_recommender)
    time_on_test = time.time() - time_on_test

    baseline_dict = {
        'MAE': MAE,
        'RMSE': RMSE,
        'time_on_test': time_on_test,
    }
    output_dataIO.save_data('baseline', baseline_dict)

    return comm_recommender


def evaluate_recommender(urm_train_last_test, urm_test, ucm, icm, communities, recommenders, output_folder_path=None, recommender_name=None,
                         n_iter=None):
    print(f'Evaluating {recommender_name} on the result of community detection.')

    recommender = CommunityDetectionAdaptiveClustering(urm_train_last_test, ucm, icm, CRITERION, communities=communities.iter(n_iter))

    evaluator_test = AC_Evaluator(urm_test)
    time_on_test = time.time()
    MAE, RMSE, MAE_all, RMSE_all = evaluator_test.evaluateRecommender(recommender)
    time_on_test = time.time() - time_on_test

    MAE_data[f'CD_AC_{n_iter}'] = MAE
    RMSE_data[f'CD_AC_{n_iter}'] = RMSE
    result_dict = {
        'MAE': MAE,
        'RMSE': RMSE,
        'time_on_test': time_on_test,
    }
    
    if output_folder_path is not None:
        dataIO = DataIO(output_folder_path)
        dataIO.save_data(f'cd_{recommender_name}', result_dict)

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

        urm_train_last_test = merge_sparse_matrices(urm_train, urm_validation)

        for recommender in recommender_list:
            recommender_name = recommender.RECOMMENDER_NAME
            output_folder_path = f'{result_folder_path}{dataset_name}/{recommender_name}/'
            if not os.path.exists(f'{output_folder_path}baseline.zip') or not os.path.exists(
                    f'{output_folder_path}{recommender_name}_best_model_last.zip'):
                train_all_data_recommender(recommender, urm_train_last_test, urm_test, ucm, icm, dataset_name, result_folder_path)
            else:
                print(f'{recommender_name} already trained and evaluated on {dataset_name}.')
        for method in method_list:
            '''
            recommend_per_method(urm_train, urm_validation, urm_test, urm_train_last_test, ucm, icm, method, sampler_list,
                                 recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy,
                                 save_model=save_model)
            '''
            plot(urm_train, method, dataset_name, result_folder_path)
            # plot(urm_test, method, dataset_name, result_folder_path)


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

    method_folder_path = f'{folder_path}{dataset_name}/{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}/'
    output_folder_path = get_community_folder_path(method_folder_path, n_iter=-1, folder_suffix=folder_suffix)

    num_iters = communities.num_iters + 1
    for n_iter in range(num_iters):
        recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name,
                           folder_path, sampler=sampler, communities=communities, n_iter=n_iter, **kwargs)

def recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, ucm, icm, method, recommender_list, dataset_name, folder_path,
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
            '''
            for community in communities.iter(n_iter):
                comm_recommender = train_recommender_on_community(recommender, community, urm_train, urm_validation,
                                                                  urm_test, ucm, icm, dataset_name, folder_path,
                                                                  method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                                  folder_suffix=folder_suffix)
                cd_recommenders.append(comm_recommender)
                n_comm += 1
            '''
            evaluate_recommender(cd_urm, urm_test, ucm, icm, communities, cd_recommenders, output_folder_path,
                                 recommender_name, n_iter=n_iter)
        else:
            print('Recommender already trained and evaluated.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--criterion', type=int, default=50)
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
        print_result(dataset_name, True, output_folder)


if __name__ == '__main__':
    args = parse_args()
    CRITERION = args.criterion
    data_reader_classes = [MovielensSample2Reader]
    # data_reader_classes = [Movielens1MReader]
    # data_reader_classes = [Movielens100KReader, Movielens1MReader, FilmTrustReader, MovielensHetrec2011Reader,
    #                        LastFMHetrec2011Reader, FrappeReader, CiteULike_aReader, CiteULike_tReader]
    recommender_list = [CommunityDetectionAdaptiveClustering]
    # method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection, UserCommunityDetection]
    method_list = [QUBOBipartiteProjectedItemCommunityDetection]
    # method_list = [QUBOBipartiteCommunityDetection]
    sampler_list = [neal.SimulatedAnnealingSampler()]
    # sampler_list = [greedy.SteepestDescentSampler(), tabu.TabuSampler()]
    # sampler_list = [LeapHybridSampler()]
    # sampler_list = [LeapHybridSampler(), neal.SimulatedAnnealingSampler(), greedy.SteepestDescentSampler(),
                    # tabu.TabuSampler()]
    result_folder_path = './results/'
    # clean_results(result_folder_path, data_reader_classes, method_list, sampler_list)
    main(data_reader_classes, method_list, sampler_list, recommender_list, result_folder_path)
    # save_results(data_reader_classes, result_folder_path, args.alpha, args.beta)
    # save_results(data_reader_classes, result_folder_path, args.alpha)
