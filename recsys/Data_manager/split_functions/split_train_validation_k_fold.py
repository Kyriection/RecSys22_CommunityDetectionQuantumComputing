#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/04/2019

@author: Anonymous
"""

import numpy as np
import scipy.sparse as sps
from recsys.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix


def merge_sparse_matrices(*matrixs):
    num = len(matrixs)
    if num == 0:
        return None
    elif num == 1:
        return matrixs[0]

    for i in range(1, num):
        assert matrixs[i - 1].shape == matrixs[i].shape, "The matrices have different shape, they should not be merged."

    data = []
    row = []
    col = []
    for matrix in matrixs:
        matrix = matrix.tocoo()
        data.append(matrix.data)
        row.append(matrix.row)
        col.append(matrix.col)

    data = np.concatenate(data)
    row = np.concatenate(row)
    col = np.concatenate(col)
    matrix = sps.csr_matrix((data, (row, col)), shape=matrixs[0].shape)

    return matrix


def split_train_in_two_percentage_user_wise(URM_train, train_percentage = 0.1, verbose = False):
    """
    The function splits an URM in two matrices selecting the number of interactions one user at a time
    :param URM_train:
    :param train_percentage:
    :param verbose:
    :return:
    """

    assert train_percentage >= 0.0 and train_percentage<=1.0, "train_percentage must be a value between 0.0 and 1.0, provided was '{}'".format(train_percentage)

    from recsys.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

    # ensure to use csr matrix or we get big problem
    URM_train = URM_train.tocsr()


    num_users, num_items = URM_train.shape

    URM_train_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items, auto_create_col_mapper=False, auto_create_row_mapper=False)
    URM_validation_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items, auto_create_col_mapper=False, auto_create_row_mapper=False)

    user_no_item_train = 0
    user_no_item_validation = 0

    for user_id in range(URM_train.shape[0]):

        start_pos = URM_train.indptr[user_id]
        end_pos = URM_train.indptr[user_id+1]


        user_profile_items = URM_train.indices[start_pos:end_pos]
        user_profile_ratings = URM_train.data[start_pos:end_pos]
        user_profile_length = len(user_profile_items)

        n_train_items = round(user_profile_length*train_percentage)

        if n_train_items == len(user_profile_items) and n_train_items > 1:
            n_train_items -= 1

        indices_for_sampling = np.arange(0, user_profile_length, dtype=np.int)
        np.random.shuffle(indices_for_sampling)

        train_items = user_profile_items[indices_for_sampling[0:n_train_items]]
        train_ratings = user_profile_ratings[indices_for_sampling[0:n_train_items]]

        validation_items = user_profile_items[indices_for_sampling[n_train_items:]]
        validation_ratings = user_profile_ratings[indices_for_sampling[n_train_items:]]

        if len(train_items) == 0:
            if verbose: print("User {} has 0 train items".format(user_id))
            user_no_item_train += 1

        if len(validation_items) == 0:
            if verbose: print("User {} has 0 validation items".format(user_id))
            user_no_item_validation += 1


        URM_train_builder.add_data_lists([user_id]*len(train_items), train_items, train_ratings)
        URM_validation_builder.add_data_lists([user_id]*len(validation_items), validation_items, validation_ratings)

    if user_no_item_train != 0:
        print("Warning: {} ({:.2f} %) of {} users have no train items".format(user_no_item_train, user_no_item_train/num_users*100, num_users))
    if user_no_item_validation != 0:
        print("Warning: {} ({:.2f} %) of {} users have no sampled items".format(user_no_item_validation, user_no_item_validation/num_users*100, num_users))

    URM_train = URM_train_builder.get_SparseMatrix()
    URM_validation = URM_validation_builder.get_SparseMatrix()


    return URM_train, URM_validation






def split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.1):
    """
    The function splits an URM in two matrices selecting the number of interactions globally
    :param URM_all:
    :param train_percentage:
    :param verbose:
    :return:
    """

    assert train_percentage >= 0.0 and train_percentage<=1.0, "train_percentage must be a value between 0.0 and 1.0, provided was '{}'".format(train_percentage)


    from recsys.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

    num_users, num_items = URM_all.shape

    URM_train_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items, auto_create_col_mapper=False, auto_create_row_mapper=False)
    URM_validation_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items, auto_create_col_mapper=False, auto_create_row_mapper=False)


    URM_train = sps.coo_matrix(URM_all)

    indices_for_sampling = np.arange(0, URM_all.nnz, dtype=np.int)
    np.random.shuffle(indices_for_sampling)

    n_train_interactions = round(URM_all.nnz * train_percentage)

    indices_for_train = indices_for_sampling[indices_for_sampling[0:n_train_interactions]]
    indices_for_validation = indices_for_sampling[indices_for_sampling[n_train_interactions:]]


    URM_train_builder.add_data_lists(URM_train.row[indices_for_train],
                                     URM_train.col[indices_for_train],
                                     URM_train.data[indices_for_train])

    URM_validation_builder.add_data_lists(URM_train.row[indices_for_validation],
                                          URM_train.col[indices_for_validation],
                                          URM_train.data[indices_for_validation])


    URM_train = URM_train_builder.get_SparseMatrix()
    URM_validation = URM_validation_builder.get_SparseMatrix()

    URM_train = sps.csr_matrix(URM_train)
    URM_validation = sps.csr_matrix(URM_validation)

    user_no_item_train = np.sum(np.ediff1d(URM_train.indptr) == 0)
    user_no_item_validation = np.sum(np.ediff1d(URM_validation.indptr) == 0)

    if user_no_item_train != 0:
        print("Warning: {} ({:.2f} %) of {} users have no train items".format(user_no_item_train, user_no_item_train/num_users*100, num_users))
    if user_no_item_validation != 0:
        print("Warning: {} ({:.2f} %) of {} users have no sampled items".format(user_no_item_validation, user_no_item_validation/num_users*100, num_users))


    return URM_train, URM_validation


def split_train_in_k_folds_user_wise(URM_train, k: int = 5, verbose = False):
    """
    The function splits an URM in k matrices selecting the number of interactions one user at a time
    :param URM_train:
    :param k:
    :param verbose:
    :return:
    """

    # ensure to use csr matrix or we get big problem
    URM_train = URM_train.tocsr()
    num_users, num_items = URM_train.shape

    user_no_item_folds = [0] * k
    URM_folds = {}
    URM_fold_builders = [
        IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items, auto_create_col_mapper=False, auto_create_row_mapper=False)
        for i in range(k)
    ]

    for user_id in range(num_users):

        start_pos = URM_train.indptr[user_id]
        end_pos = URM_train.indptr[user_id+1]


        user_profile_items = URM_train.indices[start_pos:end_pos]
        user_profile_ratings = URM_train.data[start_pos:end_pos]
        user_profile_length = len(user_profile_items)

        indices_for_sampling = np.arange(0, user_profile_length, dtype=np.int)
        np.random.shuffle(indices_for_sampling)

        for i in range(k):
            n_fold_begin = round(i / k * user_profile_length)
            n_fold_end = round((i + 1) / k * user_profile_length)
            fold_items = user_profile_items[indices_for_sampling[n_fold_begin:n_fold_end]]
            fold_ratings = user_profile_ratings[indices_for_sampling[n_fold_begin:n_fold_end]]
            if len(fold_items) == 0:
                if verbose: print("User {} has 0 items in {}'th fold.".format(user_id, i))
                user_no_item_folds[i] += 1

            URM_fold_builders[i].add_data_lists([user_id]*len(fold_items), fold_items, fold_ratings)

    for i in range(k):
        URM_folds[str(i)] = URM_fold_builders[i].get_SparseMatrix()
        if user_no_item_folds[i] != 0:
            print("Warning: {} ({:.2f} %) of {} users have no items in {}'th fold."
                  .format(user_no_item_folds[i], user_no_item_folds[i]/num_users*100, num_users, i))

    return URM_folds


def split_train_in_k_folds_global_sample(URM_all, k: int = 5):
    """
    The function splits an URM in k matrices selecting the number of interactions globally
    :param URM_all:
    :param k:
    :return:
    """

    num_users, num_items = URM_all.shape

    URM_coo = sps.coo_matrix(URM_all)

    indices_for_sampling = np.arange(0, URM_all.nnz, dtype=np.int)
    np.random.shuffle(indices_for_sampling)

    URM_folds = {}
    for i in range(k):
        n_fold_begin = round(i / k * URM_all.nnz)
        n_fold_end = round((i + 1) / k * URM_all.nnz)
        indices_for_fold = indices_for_sampling[indices_for_sampling[n_fold_begin:n_fold_end]]
        URM_fold_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items, auto_create_col_mapper=False, auto_create_row_mapper=False)
        URM_fold_builder.add_data_lists(URM_coo.row[indices_for_fold],
                                        URM_coo.col[indices_for_fold],
                                        URM_coo.data[indices_for_fold])
        URM_fold = URM_fold_builder.get_SparseMatrix()
        URM_fold = sps.csr_matrix(URM_fold)

        user_no_item_fold = np.sum(np.ediff1d(URM_fold.indptr) == 0)
        if user_no_item_fold != 0:
            print("Warning: {} ({:.2f} %) of {} users have no train items".format(user_no_item_fold, user_no_item_fold/num_users*100, num_users))

        URM_folds[str(i)] = URM_fold

    return URM_folds

