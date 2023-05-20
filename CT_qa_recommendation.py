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
    TMPCD, QUBOLongTailCommunityDetection
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, \
    MovielensSampleReader, MovielensSample2Reader
# from recsys.Evaluation.Evaluator import EvaluatorHoldout
from recsys.Evaluation.EvaluatorSeparate import EvaluatorSeparate
from recsys.Recommenders.BaseRecommender import BaseRecommender
from recsys.Recommenders.LRRecommender import LRRecommender
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.urm import get_community_urm, load_data, merge_sparse_matrices
from utils.plot import plot_line, plot_scatter
from utils.derived_variables import create_derived_variables
# from results.read_results import print_result
from results.plot_results import print_result

logging.basicConfig(level=logging.INFO)
CUT_RATIO: float = None
PLOT_CUT = 30
MIN_RATING_NUM = 1
TOTAL_DATA = {}
EI: bool = False # EI if True else TC or CT

def plot(urm, output_folder_path, n_iter, result_df):
    global MIN_RATING_NUM, PLOT_CUT
    C_quantity = np.ediff1d(urm.tocsr().indptr) # count of each row
    data: np.ndarray = result_df.values # [MAE, MSE, num_rating]
    # delete users whose test rating num < MIN_RATING_NUM
    ignore_users = data[:, 2] < MIN_RATING_NUM
    data = data[~ignore_users]
    C_quantity = C_quantity[~ignore_users]
    n_users = C_quantity.size
    # sort by train rating num
    data = data[np.argsort(C_quantity)]
    C_quantity = np.sort(C_quantity)

    # plot by item rank
    x = range(n_users)
    MAE_data = dict(TC_qa = [mae for mae, mse, num_rating in data])
    RMSE_data = dict(TC_qa = [np.sqrt(mse) for mae, mse, num_rating in data])
    plot_scatter(x, MAE_data, output_folder_path, 'item rank', 'MAE')
    plot_scatter(x, RMSE_data, output_folder_path, 'item rank', 'RMSE')
    # cluster by C_quantity
    tot_mae = 0.0
    tot_rmse = 0.0
    tot_num_rating = 0
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
        tot_mae += mae * num_rating
        tot_rmse += mse * num_rating
        tot_num_rating += num_rating
        cluster_data[quantity] = _data
    # BTY, dict.items() == zip(dict.keys(), dict.values())
    x = sorted(list(cluster_data.keys()))
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
    # print tot
    tot_mae = round(tot_mae / tot_num_rating, 4)
    tot_rmse = round(np.sqrt(tot_rmse / tot_num_rating), 4)
    TOTAL_DATA[n_iter] = dict(MAE=tot_mae, RMSE=tot_rmse)
    print(f'n_iter:{n_iter}, Total MAE = {tot_mae}, Total RMSE = {tot_rmse}')

    # cluster to PLOT_CUT points
    '''
    def get_cut_size(cut_num: int):
        l = 1
        r = urm.size
        while l < r:
            mid = (l + r + 1) // 2
            cnt = 0
            cur = 0
            for key in cluster_data:
                cur += cluster_data[key][2]
                if cur >= mid:
                    cur = 0
                    cnt += 1
            if cur > 0:
                cnt += 1
            if cnt >= cut_num:
                l = mid
            else:
                r = mid - 1
        return l

    cut_size = get_cut_size(PLOT_CUT)
    x_ = x
    x = []
    MAE_data = []
    RMSE_data = []
    MAE = 0.0
    MSE = 0.0
    num_rating = 0
    for key in x_:
        data = cluster_data[key]
        MAE += _data[0]
        MSE += _data[1]
        num_rating += _data[2]
        if num_rating >= cut_size:
            x.append(key)
            MAE_data.append(MAE / num_rating)
            RMSE_data.append(np.sqrt(MSE / num_rating))
            MAE = 0.0
            MSE = 0.0
            num_rating = 0
    if num_rating > 0:
        x.append(x_[-1])
        MAE_data.append(MAE / num_rating)
        RMSE_data.append(np.sqrt(MSE / num_rating))
    logging.info(f'cut_size={cut_size}, len(x)={len(x)}')
    '''
    cut_points = np.arange(1, PLOT_CUT + 1) * (n_users // PLOT_CUT)
    cut_points[-1] = n_users - 1
    x = C_quantity[cut_points]
    x = sorted(set(x))
    MAE_data = []
    RMSE_data = []
    cd_key = list(cluster_data.keys())
    cd_i = 0
    for cut_quantity in x:
        MAE = 0.0
        MSE = 0.0
        num_rating = 0
        while cd_i < len(cd_key) and cd_key[cd_i] <= cut_quantity:
            _data = cluster_data[cd_key[cd_i]]
            MAE += _data[0]
            MSE += _data[1]
            num_rating += _data[2]
            cd_i += 1
        MAE_data.append(MAE / num_rating)
        RMSE_data.append(np.sqrt(MSE / num_rating))
    MAE_data = dict(TC_qa = MAE_data)
    RMSE_data = dict(TC_qa = RMSE_data)
    plot_line(x, MAE_data, output_folder_path, 'the number of ratings (clustered)', 'MAE')
    plot_line(x, RMSE_data, output_folder_path, 'the number of ratings (clustered)', 'RMSE')


def head_tail_cut(urm_train, urm_validation, urm_test, icm, ucm):
    '''
    return (head)urm_train, urm_validation, urm_test, icm, ucm,\
           (tail)urm_train, urm_validation, urm_test, icm, ucm
    '''
    n_users, n_items = urm_train.shape
    C_quantity = np.ediff1d(urm_train.tocsr().indptr) # count of each row
    cut_quantity = sorted(C_quantity, reverse=True)[int(len(C_quantity) * CUT_RATIO)]
    head_user_mask = C_quantity > cut_quantity
    # tail_user_mask = C_quantity <= cut_quantity
    communities = Communities(head_user_mask, np.ones(n_items).astype(bool))
    tail_community, head_community = communities.c0, communities.c1
    logging.info(f'head tail cut at {cut_quantity}, head size: {len(head_community.users)}, tail size: {len(tail_community.users)}')
    t_urm_train, _, _, t_icm, t_ucm = get_community_urm(urm_train, community=tail_community, filter_items=False, remove=True, icm=icm, ucm=ucm)
    t_urm_validation, _, _ = get_community_urm(urm_validation, community=tail_community, filter_items=False, remove=True)
    t_urm_test, _, _ = get_community_urm(urm_test, community=tail_community, filter_items=False, remove=True)
    h_urm_train, _, _, h_icm, h_ucm = get_community_urm(urm_train, community=head_community, filter_items=False, remove=True, icm=icm, ucm=ucm)
    h_urm_validation, _, _ = get_community_urm(urm_validation, community=head_community, filter_items=False, remove=True)
    h_urm_test, _, _ = get_community_urm(urm_test, community=head_community, filter_items=False, remove=True)
    return h_urm_train, h_urm_validation, h_urm_test, h_icm, h_ucm, \
           t_urm_train, t_urm_validation, t_urm_test, t_icm, t_ucm


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

    evaluator_test = EvaluatorSeparate(urm_test) # TODO

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
    print(f'Training {recommender_name} on community {n_comm if n_comm is not None else ""} of iteration {n_iter}...')
    logging.info(f"community.size({len(community.users)}, {len(community.items)})")
    logging.info(f"community.shape({len(community.user_mask)}, {len(community.item_mask)})")
    logging.debug(f'urm.shape: {urm_train.shape}')

    output_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                   folder_suffix=folder_suffix)
    output_folder_path = f'{output_folder_path}{recommender_name}/'

    base_recommender_path = f'{results_folder_path}{dataset_name}/{recommender_name}/'

    c_urm_train, _, _, c_icm, c_ucm = get_community_urm(urm_train, community=community, filter_items=False, icm=icm, ucm=ucm)
    c_urm_validation, _, _ = get_community_urm(urm_validation, community=community, filter_items=False)
    c_urm_test, _, _ = get_community_urm(urm_test, community=community, filter_items=False)
    c_urm_train_last_test = merge_sparse_matrices(c_urm_train, c_urm_validation)

    ignore_users = np.arange(c_urm_train_last_test.shape[0])[np.logical_not(community.user_mask)]
    evaluator_validation = EvaluatorSeparate(c_urm_validation, ignore_users=ignore_users)
    evaluator_test = EvaluatorSeparate(c_urm_test, ignore_users=ignore_users)

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
                         n_iter=None):
    print(f'Evaluating {recommender_name} on the result of community detection.')

    recommender = CommunityDetectionRecommender(urm_train_last_test, communities=communities, recommenders=recommenders,
                                                n_iter=n_iter)

    evaluator_test = EvaluatorSeparate(urm_test)
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
    global EI
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
            head_tail_cut(urm_train, urm_validation, urm_test, icm, ucm)
        head_flag = h_urm_train.shape[0] > 0
        logging.info(f'head shape: {h_urm_train.shape}, tail shape: {t_urm_train.shape}')

        t_urm_train_last_test = merge_sparse_matrices(t_urm_train, t_urm_validation)
        if head_flag:
            h_urm_train_last_test = merge_sparse_matrices(h_urm_train, h_urm_validation)

        for recommender in recommender_list:
            recommender_name = recommender.RECOMMENDER_NAME
            output_folder_path = f'{result_folder_path}{dataset_name}/{recommender_name}/'
            if not os.path.exists(f'{output_folder_path}baseline.zip') or not os.path.exists(
                    f'{output_folder_path}{recommender_name}_best_model_last.zip'):
                train_all_data_recommender(recommender, t_urm_train_last_test, t_urm_test, t_ucm, t_icm, dataset_name, result_folder_path)
            else:
                print(f'{recommender_name} already trained and evaluated on {dataset_name}.')
        for method in method_list:
            recommend_per_method(t_urm_train, t_urm_validation, t_urm_test, t_urm_train_last_test, t_ucm, t_icm, method, sampler_list,
                                 recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy,
                                 save_model=save_model, each_item=EI)
            if not head_flag or EI:
                return
            recommend_per_method(h_urm_train, h_urm_validation, h_urm_test, h_urm_train_last_test, h_ucm, h_icm, method, sampler_list,
                                 recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy,
                                 save_model=save_model, each_item=True)
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
    parser.add_argument('-c', '--cut_ratio', type=float, default=0.0)
    parser.add_argument('-a', '--alpha', type=float, default=1.0)
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


def save_results(data_reader_classes, result_folder_path, method_list, *args):
    global CUT_RATIO
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
        print_result(CUT_RATIO, data_reader, method_list, False, output_folder)


if __name__ == '__main__':
    args = parse_args()
    CUT_RATIO = args.cut_ratio
    data_reader_classes = [MovielensSample2Reader]
    # data_reader_classes = [Movielens1MReader]
    # data_reader_classes = [Movielens100KReader, Movielens1MReader, FilmTrustReader, MovielensHetrec2011Reader,
    #                        LastFMHetrec2011Reader, FrappeReader, CiteULike_aReader, CiteULike_tReader]
    recommender_list = [LRRecommender]
    # method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection]
    method_list = [HybridCommunityDetection]
    method_list = [QUBOBipartiteCommunityDetection]
    # method_list = [QUBOLongTailCommunityDetection]
    sampler_list = [neal.SimulatedAnnealingSampler()]
    # sampler_list = [greedy.SteepestDescentSampler(), tabu.TabuSampler()]
    # sampler_list = [LeapHybridSampler()]
    # sampler_list = [LeapHybridSampler(), neal.SimulatedAnnealingSampler(), greedy.SteepestDescentSampler(),
                    # tabu.TabuSampler()]
    result_folder_path = './results/'
    clean_results(result_folder_path, data_reader_classes, method_list, sampler_list, recommender_list)
    main(data_reader_classes, method_list, sampler_list, recommender_list, result_folder_path)
    save_results(data_reader_classes, result_folder_path, method_list, args.cut_ratio, args.alpha)
