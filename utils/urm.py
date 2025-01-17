import logging

import numpy as np
import scipy.sparse as sps

from CommunityDetection import Community, Communities
from recsys.Data_manager.DataSplitter import DataSplitter as _DataSplitter
from recsys.Data_manager.DataSplitter_Holdout import DataSplitter_Holdout
from recsys.Data_manager.DataSplitter_k_fold import DataSplitter_k_fold
from recsys.Recommenders.Recommender_utils import reshapeSparse
from utils.derived_variables import normalization


def load_icm_ucm(tag: str, data_splitter: _DataSplitter, n_users: int, n_items: int):
    try:
        if tag == 'cluster':
            icm = data_splitter.get_ICM_from_name('ICM_one_hot')
        elif tag == 'recommend':
            icm = data_splitter.get_ICM_from_name('ICM_all')
    except Exception:
        logging.warning('Load ICM_all Faild.')
        icm = sps.csr_matrix(([], ([], [])), shape=(n_items, 1))
        icm = normalization(icm)
    try:
        ucm = data_splitter.get_UCM_from_name('UCM_all')
    except Exception:
        logging.warning('Load UCM_all Faild.')
        ucm = sps.csr_matrix(([], ([], [])), shape=(n_users, 1))
        ucm = normalization(ucm)
    return icm, ucm


def load_data(data_reader, split_quota=None, user_wise=True, make_implicit=True, threshold=None, icm_ucm=False):
    """
    return urm_train, urm_validation, urm_test(, icm, ucm)
    """
    print('Loading data...')

    if split_quota is None:
        split_quota = [70, 10, 20]

    data_splitter = DataSplitter_Holdout(data_reader, split_quota, user_wise=user_wise)
    data_splitter.load_data()

    urm_train, urm_validation, urm_test = data_splitter.get_holdout_split()

    if make_implicit:
        urm_train = explicit_to_implicit_urm(urm_train, threshold=threshold)
        urm_validation = explicit_to_implicit_urm(urm_validation, threshold=threshold)
        urm_test = explicit_to_implicit_urm(urm_test, threshold=threshold)

    if icm_ucm:
        n_users, n_items = urm_train.shape
        icm, ucm = load_icm_ucm(data_splitter, n_users, n_items)
        return urm_train, urm_test, icm, ucm
    else:
        return urm_train, urm_validation, urm_test  # , var_mapping


def load_data_k_fold(tag: str, data_reader, user_wise=True, make_implicit=True, threshold=None, icm_ucm=False, n_folds: int = 5, k: int = 0):
    """
    return urm_train, urm_test(, icm, ucm)
    """
    print('Loading data...')

    data_splitter = DataSplitter_k_fold(data_reader, user_wise=user_wise, n_folds=n_folds)
    data_splitter.load_data()

    urm_train, urm_test = data_splitter.get_holdout_split(k=k)

    if make_implicit:
        urm_train = explicit_to_implicit_urm(urm_train, threshold=threshold)
        urm_test = explicit_to_implicit_urm(urm_test, threshold=threshold)

    if icm_ucm:
        n_users, n_items = urm_train.shape
        icm, ucm = load_icm_ucm(tag, data_splitter, n_users, n_items)
        return urm_train, urm_test, icm, ucm
    else:
        return urm_train, urm_test  # , var_mapping


def merge_sparse_matrices(matrix_a, matrix_b):
    assert matrix_a.shape == matrix_b.shape, "The two matrices have different shape, they should not be merged."

    matrix_a = matrix_a.tocoo()
    matrix_b = matrix_b.tocoo()

    data_a = matrix_a.data
    row_a = matrix_a.row
    col_a = matrix_a.col

    data_b = matrix_b.data
    row_b = matrix_b.row
    col_b = matrix_b.col

    data = np.concatenate((data_a, data_b))
    row = np.concatenate((row_a, row_b))
    col = np.concatenate((col_a, col_b))

    matrix = sps.coo_matrix((data, (row, col)), shape=matrix_a.shape)

    n_users = max(matrix_a.shape[0], matrix_b.shape[0])
    n_items = max(matrix_a.shape[1], matrix_b.shape[1])
    new_shape = (n_users, n_items)

    matrix = reshapeSparse(matrix, new_shape)

    return matrix


def explicit_to_implicit_urm(urm, threshold=None):
    urm_data = urm.data

    if threshold is not None:
        urm_data_mask = urm_data >= threshold
        urm_data = urm_data_mask.astype(int)
    else:
        urm_data = np.ones_like(urm_data)

    urm.data = urm_data
    urm.eliminate_zeros()
    return urm


def get_community_urm(urm, community: Community, filter_users=True, filter_items=True, remove=False, icm=None, ucm=None):
    new_urm = urm.copy()
    new_icm = icm.copy() if icm is not None else None
    new_ucm = ucm.copy() if ucm is not None else None
    n_users, n_items = urm.shape
    new_users = np.arange(n_users)
    new_items = np.arange(n_items)

    if filter_users:
        users = community.user_mask
        new_users = new_users[users]
        if remove:
            new_urm = new_urm[users, :]
            if ucm is not None:
                new_ucm = new_ucm[users, :]
        else:
            users = np.logical_not(users)
            new_urm[users, :] = 0
            if ucm is not None:
                new_ucm[users, :] = 0

    if filter_items:
        items = community.item_mask
        new_items = new_items[items]
        if remove:
            new_urm = new_urm[:, items]
            if icm is not None:
                new_icm = new_icm[items, :]
        else:
            items = np.logical_not(items)
            new_urm[:, items] = 0
            if icm is not None:
                new_icm[items, :] = 0

    new_urm.eliminate_zeros()
    if new_icm is not None:
        new_icm.eliminate_zeros()
    if new_ucm is not None:
        new_ucm.eliminate_zeros()
    if icm is not None or ucm is not None:
        return new_urm, new_users, new_items, new_icm, new_ucm
    else:
        return new_urm, new_users, new_items


def show_urm_info(urm):
    def show_quantity_info(quantity):
        min_val = np.min(quantity)
        max_val = np.max(quantity)
        mean_val = np.mean(quantity)
        var_val = np.var(quantity)
        print(f'min={min_val}, max={max_val}, mean={mean_val}, variance={var_val}')

    print('--------------- show urm info ----------------')
    C_quantity = np.ediff1d(urm.tocsr().indptr) # count of each row
    I_quantity = np.ediff1d(urm.tocsc().indptr) # count of each colum
    print(f'urm.shape={urm.shape}, urm.size={urm.size}')
    print('user info')
    show_quantity_info(C_quantity)
    print('item info')
    show_quantity_info(I_quantity)
    print('----------------------------------------------')


def head_tail_cut(cut_ratio, urm_train, urm_validation, urm_test, icm, ucm):
    '''
    return (head)urm_train, urm_validation, urm_test, icm, ucm,\
           (tail)urm_train, urm_validation, urm_test, icm, ucm
    '''
    urm_train_last_test = merge_sparse_matrices(urm_train, urm_validation)
    urm_all = merge_sparse_matrices(urm_train_last_test, urm_test)
    n_users, n_items = urm_all.shape
    C_quantity = np.ediff1d(urm_all.tocsr().indptr) # count of each row
    cut_quantity = sorted(C_quantity, reverse=True)[int(len(C_quantity) * cut_ratio)]
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


def head_tail_cut_k_fold(cut_ratio, urm_train, urm_test, icm, ucm):
    '''
    return (head)urm_train, urm_test, icm, ucm,
           (tail)urm_train, urm_test, icm, ucm
    '''
    urm_all = merge_sparse_matrices(urm_train, urm_test)
    n_users, n_items = urm_all.shape
    C_quantity = np.ediff1d(urm_all.tocsr().indptr) # count of each row
    cut_quantity = sorted(C_quantity, reverse=True)[int(len(C_quantity) * cut_ratio)]
    head_user_mask = C_quantity > cut_quantity
    # tail_user_mask = C_quantity <= cut_quantity
    communities = Communities(head_user_mask, np.ones(n_items).astype(bool))
    tail_community, head_community = communities.c0, communities.c1
    logging.info(f'head tail cut at {cut_quantity}, head size: {len(head_community.users)}, tail size: {len(tail_community.users)}')
    t_urm_train, _, _, t_icm, t_ucm = get_community_urm(urm_train, community=tail_community, filter_items=False, remove=True, icm=icm, ucm=ucm)
    t_urm_test, _, _ = get_community_urm(urm_test, community=tail_community, filter_items=False, remove=True)
    h_urm_train, _, _, h_icm, h_ucm = get_community_urm(urm_train, community=head_community, filter_items=False, remove=True, icm=icm, ucm=ucm)
    h_urm_test, _, _ = get_community_urm(urm_test, community=head_community, filter_items=False, remove=True)
    return h_urm_train, h_urm_test, h_icm, h_ucm, \
           t_urm_train, t_urm_test, t_icm, t_ucm
