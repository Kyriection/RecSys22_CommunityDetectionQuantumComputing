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
    SpectralClustering, QUBOBipartiteProjectedItemCommunityDetection, CommunitiesEI
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, \
    MovielensSampleReader, MovielensSample2Reader
# from recsys.Evaluation.Evaluator import EvaluatorHoldout
from recsys.Evaluation.EvaluatorSeparate import EvaluatorSeparateHoldout
from recsys.Recommenders.BaseRecommender import BaseRecommender
from recsys.Recommenders.LRRecommender import LRRecommender
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.urm import get_community_urm, load_data, merge_sparse_matrices
from utils.plot import plot_line, plot_scatter
from utils.derived_variables import create_derived_variables
from results.read_results import print_result

logging.basicConfig(level=logging.INFO)
CRITERION: int = None
EI: bool = False
MAE_data = {}
RMSE_data = {}
PLOT_CUT = 30

def plot(urm, output_folder_path, n_iter, result_df):
# def plot(urm, method, dataset_name, folder_path):
    C_quantity = np.ediff1d(urm.tocsr().indptr) # count of each row
    n_users = C_quantity.size
    data: list = result_df.values.tolist() # [MAE, MSE, num_rating]
    data = [x for _, x in sorted(zip(C_quantity, data))]
    C_quantity = sorted(C_quantity)
    # plot by item rank
    x = range(n_users)
    MAE_data = dict(TC_qa = [mae for mae, mse, num_rating in data])
    RMSE_data = dict(TC_qa = [np.sqrt(mse) for mae, mse, num_rating in data])
    plot_scatter(x, MAE_data, output_folder_path, 'item rank', 'MAE')
    plot_scatter(x, RMSE_data, output_folder_path, 'item rank', 'RMSE')

    cluster_data = {}
    for i in range(n_users):
        quantity = C_quantity[i]
        mae, mse, num_rating = data[i]
        if num_rating == 0:
            continue
        _data = cluster_data.get(quantity, [0.0, 0.0, 0])
        _data[0] += mae * num_rating
        _data[1] += mse * num_rating
        _data[2] += num_rating
        cluster_data[quantity] = _data
    # BTY, dict.items() == zip(dict.keys(), dict.values())
    x = list(cluster_data.keys())
    MAE_data = []
    RMSE_data = []
    for key in x:
        mae, mse, num_rating = cluster_data[key]
        MAE_data.append(mae / num_rating)
        RMSE_data.append(np.sqrt(mse / num_rating))
    MAE_data = dict(TC_qa = MAE_data)
    RMSE_data = dict(TC_qa = RMSE_data)
    # plot by #ratings
    plot_scatter(x, MAE_data, output_folder_path, 'the number of ratings', 'MAE')
    plot_scatter(x, RMSE_data, output_folder_path, 'the number of ratings', 'RMSE')

    '''
    max_x = I_quantity[-1]
    gap = max((max_x + PLOT_CUT - 1)  // PLOT_CUT, max_x // (PLOT_CUT - 1)) # max_x // gap = PLOT_CUT - 1
    # print(f'max_x={max(x)}, gap={gap}, max_quantity={I_quantity[-1]}')
    for data in (MAE_data, RMSE_data):
        for key in data:
            new_data = [0.0] * PLOT_CUT
            cnt = [0] * PLOT_CUT
            assert len(data[key]) == len(x), f'[Error] len_data({len(data[key])}) != len_x({len(x)})'
            for i, quantity in enumerate(x):
                j = quantity // gap
                assert j < PLOT_CUT, f'[Error] quantity // gap >= PLOT_CUT, {quantity} // {gap} = {quantity // gap} >= {PLOT_CUT}'
                new_data[j] += data[key][i]
                cnt[j] += 1
            data[key] = []
            for i in range(PLOT_CUT):
                if cnt[i]:
                    data[key].append(new_data[i] / cnt[i])
                else:
                    data[key].append(data[key][-1] if data[key] else 0.0)
    # plot by #ratings(cluster)
    x = np.arange(PLOT_CUT) * gap
    plot_line(x, MAE_data, output_folder_path, f'the number of ratings (cluster {gap})', 'MAE')
    plot_line(x, RMSE_data, output_folder_path, f'the number of ratings (cluster {gap})', 'RMSE')
    '''


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
    recommender_name = recommender.RECOMMENDER_NAME
    output_folder_path = f'{results_folder_path}{dataset_name}/{recommender_name}/'

    evaluator_test = EvaluatorSeparateHoldout(urm_test) # TODO

    print(f'Training {recommender_name} on all data...')

    time_on_train = time.time()
    rec = recommender(urm_train_last_test, ucm, icm)
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


def train_recommender_on_community(recommender, community, urm_train, urm_validation, urm_test, ucm, icm, dataset_name,
                                   results_folder_path, method_folder_path, n_iter=0, n_comm=None, folder_suffix='',
                                   **kwargs):
    """
    validation: fit(train), predict(validation)
    test: fit(train + validation), predict(test)
    baseline: best_model fit(train + validation), predict(test)
    """
    recommender_name = recommender.RECOMMENDER_NAME
    logging.info(f'len(community.users)={len(community.users)}')
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
    evaluator_validation = EvaluatorSeparateHoldout(c_urm_validation, ignore_users=ignore_users)
    evaluator_test = EvaluatorSeparateHoldout(c_urm_test, ignore_users=ignore_users)

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
    base_evaluator_test = EvaluatorSeparateHoldout(c_urm_test, ignore_users=ignore_users)

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
                         n_iter=None):
    print(f'Evaluating {recommender_name} on the result of community detection.')

    recommender = CommunityDetectionRecommender(urm_train_last_test, communities=communities, recommenders=recommenders,
                                                n_iter=n_iter)

    evaluator_test = EvaluatorSeparateHoldout(urm_test)
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

    plot(urm_train_last_test, output_folder_path, n_iter, result_df)
    return result_dict


def normalization(variables):
    min_val = np.min(variables, axis=0)
    max_val = np.max(variables, axis=0)
    _range = max_val - min_val
    _range[_range <= 0] = 1
    return (variables - min_val) / _range


def create_related_variables(urm, icm, ucm):
    C_aver_rating, C_quantity, C_seen_popularity, C_seen_rating,\
    I_aver_rating, I_quantity, I_likability = create_derived_variables(urm)
    item_related_variables = np.hstack([
        I_aver_rating.reshape((-1, 1)),
        I_quantity.reshape((-1, 1)),
        # I_likability.reshape((-1, 1)),
        icm.toarray(),
    ])
    user_related_variables = np.hstack([
        C_aver_rating.reshape((-1, 1)),
        C_quantity.reshape((-1, 1)),
        # C_seen_popularity.reshape((-1, 1)),
        # C_seen_rating.reshape((-1, 1)),
        ucm.toarray(),
    ])
    item_related_variables = normalization(item_related_variables)
    user_related_variables = normalization(user_related_variables)
    return item_related_variables, user_related_variables


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
            recommend_per_method(urm_train, urm_validation, urm_test, urm_train_last_test, ucm, icm, method, sampler_list,
                                 recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy,
                                 save_model=save_model, each_item=False)
            # plot(urm_train, method, dataset_name, result_folder_path)
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
    parser.add_argument('-c', '--criterion', type=int, default=50)
    args = parser.parse_args()
    return args

def clean_results(result_folder_path, data_reader_classes, method_list, sampler_list, recommender_list):
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
        '''
        for recommender in recommender_list:
            recommender_folder_path = os.path.join(dataset_folder_path, recommender.RECOMMENDER_NAME)
            logging.debug(f'clean {recommender_folder_path}')
            if os.path.exists(recommender_folder_path):
                shutil.rmtree(recommender_folder_path)
        '''
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
    recommender_list = [LRRecommender]
    method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection]
    # method_list = [QUBOBipartiteCommunityDetection]
    sampler_list = [neal.SimulatedAnnealingSampler()]
    # sampler_list = [greedy.SteepestDescentSampler(), tabu.TabuSampler()]
    # sampler_list = [LeapHybridSampler()]
    # sampler_list = [LeapHybridSampler(), neal.SimulatedAnnealingSampler(), greedy.SteepestDescentSampler(),
                    # tabu.TabuSampler()]
    result_folder_path = './results/'
    clean_results(result_folder_path, data_reader_classes, method_list, sampler_list, recommender_list)
    main(data_reader_classes, method_list, sampler_list, recommender_list, result_folder_path)
    # save_results(data_reader_classes, result_folder_path, args.alpha, args.beta)
    # save_results(data_reader_classes, result_folder_path, args.alpha)