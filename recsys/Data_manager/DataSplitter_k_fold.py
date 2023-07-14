#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/01/18

@author: Anonymous
"""

import scipy.sparse as sps
import numpy as np
from utils.DataIO import DataIO

from recsys.Data_manager.DataSplitter import DataSplitter as _DataSplitter
from recsys.Data_manager.DataReader import DataReader as _DataReader
from recsys.Data_manager.split_functions.split_train_validation_k_fold import split_train_in_k_folds_user_wise,\
      split_train_in_k_folds_global_sample, merge_sparse_matrices
from recsys.Data_manager.DataReader_utils import compute_density, reconcile_mapper_with_removed_tokens
from recsys.Data_manager.data_consistency_check import assert_disjoint_matrices, assert_URM_ICM_mapper_consistency





class DataSplitter_k_fold(_DataSplitter):
    """
    The splitter creates a random holdout of three split: train, validation and test
    The split is performed user-wise
    """

    """
     - It exposes the following functions
        - load_data(save_folder_path = None, force_new_split = False)   loads the data or creates a new split
    
    
    """

    DATA_SPLITTER_NAME = "DataSplitter_k_fold"

    SPLIT_ICM_DICT = None
    SPLIT_UCM_DICT = None
    SPLIT_ICM_MAPPER_DICT = None
    SPLIT_UCM_MAPPER_DICT = None
    SPLIT_GLOBAL_MAPPER_DICT = None
    FOLD_URM_LIST = None



    def __init__(self, dataReader_object:_DataReader, user_wise = True, allow_cold_users = False,
                 forbid_new_split = False, force_new_split = False, n_folds: int = 5):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        """

        self.n_folds = n_folds
        self.allow_cold_users = allow_cold_users
        self.user_wise = user_wise

        super(DataSplitter_k_fold, self).__init__(dataReader_object, forbid_new_split=forbid_new_split, force_new_split=force_new_split)



    def _get_split_subfolder_name(self):
        """

        :return: warm_{n_folds}_fold/
        """

        if self.user_wise:
            user_wise_string = "user_wise"
        else:
            user_wise_string = "global_sample"

        return "{}_fold_{}/".format(self.n_folds, user_wise_string)



    def get_statistics_URM(self):

        self._assert_is_initialized()

        n_users, n_items = self.FOLD_URM_LIST['0'].shape

        statistics_string = "DataReader: {}\n" \
                            "\tNum items: {}\n" \
                            "\tNum users: {}\n".format(
            self.dataReader_object._get_dataset_name(),
            n_items,
            n_users,
        )

        for i in range(self.n_folds):
            statistics_string += "\tFold {} \tinteractions {}, \tdensity {:.2E}\n".format(
                i, self.FOLD_URM_LIST[str(i)].nnz, compute_density(self.FOLD_URM_LIST[str(i)])
            )


        self._print(statistics_string)

        print("\n")




    def get_ICM_from_name(self, ICM_name):
        return self.SPLIT_ICM_DICT[ICM_name].copy()

    def get_UCM_from_name(self, UCM_name):
        return self.SPLIT_UCM_DICT[UCM_name].copy()

    def get_statistics_ICM(self):

        self._assert_is_initialized()

        if len(self.dataReader_object.get_loaded_ICM_names())>0:

            for ICM_name, ICM_object in self.SPLIT_ICM_DICT.items():

                n_items, n_features = ICM_object.shape

                statistics_string = "\tICM name: {}, Num features: {}, feature occurrences: {}, density {:.2E}".format(
                    ICM_name,
                    n_features,
                    ICM_object.nnz,
                    compute_density(ICM_object)
                )

                print(statistics_string)


    def get_statistics_UCM(self):

        self._assert_is_initialized()

        if len(self.dataReader_object.get_loaded_UCM_names())>0:

            for UCM_name, UCM_object in self.SPLIT_UCM_DICT.items():

                n_items, n_features = UCM_object.shape

                statistics_string = "\tUCM name: {}, Num features: {}, feature occurrences: {}, density {:.2E}".format(
                    UCM_name,
                    n_features,
                    UCM_object.nnz,
                    compute_density(UCM_object)
                )

                print(statistics_string)


    def _assert_is_initialized(self):
         assert self.FOLD_URM_LIST is not None, "{}: Unable to load data split. The split has not been generated yet, call the load_data function to do so.".format(self.DATA_SPLITTER_NAME)


    def get_holdout_split(self, k: int = 0):
        """
        :return: URM_train, URM_test
        """

        self._assert_is_initialized()
        assert k < self.n_folds, f'get_holdout_split failed: k={k} should be less than n_folds={self.n_folds}.'

        URM_train = []
        URM_test = self.FOLD_URM_LIST[str(k)].copy()
        
        for i in range(self.n_folds):
            if i != k:
                URM_train.append(self.FOLD_URM_LIST[str(i)])
        URM_train = merge_sparse_matrices(*URM_train)

        return URM_train, URM_test


    def _split_data_from_original_dataset(self, save_folder_path):


        self.loaded_dataset = self.dataReader_object.load_data()
        self._load_from_DataReader_ICM_and_mappers(self.loaded_dataset)

        URM_all = self.loaded_dataset.get_URM_all()

        if self.user_wise:
            # URM_train_validation, URM_test = split_train_in_two_percentage_user_wise(URM_all, train_percentage = train_quota + validation_quota)
            self.FOLD_URM_LIST = split_train_in_k_folds_user_wise(URM_all, self.n_folds)
        else:
            # URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = train_quota + validation_quota)
            self.FOLD_URM_LIST = split_train_in_k_folds_global_sample(URM_all, self.n_folds)


        if not self.allow_cold_users:

            user_interactions = np.ediff1d(URM_all.indptr)
            user_to_preserve = user_interactions >= 1
            user_to_remove = np.logical_not(user_to_preserve)

            n_users = URM_all.shape[0]

            if user_to_remove.sum() >0 :

                self._print("Removing {} ({:.2f} %) of {} users because they have no interactions in train data.".format(user_to_remove.sum(), user_to_remove.sum()/n_users*100, n_users))

                for key in self.FOLD_URM_LIST:
                    self.FOLD_URM_LIST[key] = self.FOLD_URM_LIST[key][user_to_preserve,:]

                self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = reconcile_mapper_with_removed_tokens(self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"],
                                                                                                                  np.arange(0, len(user_to_remove), dtype=np.int)[user_to_remove])

                for UCM_name, UCM_object in self.SPLIT_UCM_DICT.items():
                    UCM_object = UCM_object[user_to_preserve,:]
                    self.SPLIT_UCM_DICT[UCM_name] = UCM_object


        self._save_split(save_folder_path)

        self._print("Split complete")


    def _save_split(self, save_folder_path):

        if save_folder_path:

            if self.allow_cold_users:
                allow_cold_users_suffix = "allow_cold_users"

            else:
                allow_cold_users_suffix = "only_warm_users"

            if self.user_wise:
                user_wise_string = "user_wise"
            else:
                user_wise_string = "global_sample"


            name_suffix = "_{}_{}".format(allow_cold_users_suffix, user_wise_string)


            split_parameters_dict = {
                            "n_folds": self.n_folds,
                            "allow_cold_users": self.allow_cold_users
                            }

            dataIO = DataIO(folder_path = save_folder_path)

            dataIO.save_data(data_dict_to_save = split_parameters_dict,
                             file_name = "split_parameters" + name_suffix)

            dataIO.save_data(data_dict_to_save = self.SPLIT_GLOBAL_MAPPER_DICT,
                             file_name = "split_mappers" + name_suffix)

            dataIO.save_data(data_dict_to_save = self.FOLD_URM_LIST,
                             file_name = "fold_URM" + name_suffix)

            if len(self.SPLIT_ICM_DICT)>0:
                dataIO.save_data(data_dict_to_save = self.SPLIT_ICM_DICT,
                                 file_name = "split_ICM" + name_suffix)

                dataIO.save_data(data_dict_to_save = self.SPLIT_ICM_MAPPER_DICT,
                                 file_name = "split_ICM_mappers" + name_suffix)


            if len(self.SPLIT_UCM_DICT)>0:
                dataIO.save_data(data_dict_to_save = self.SPLIT_UCM_DICT,
                                 file_name = "split_UCM" + name_suffix)

                dataIO.save_data(data_dict_to_save = self.SPLIT_UCM_MAPPER_DICT,
                                 file_name = "split_UCM_mappers" + name_suffix)



    def _load_previously_built_split_and_attributes(self, save_folder_path):
        """
        Loads all URM and ICM
        :return:
        """

        if self.allow_cold_users:
            allow_cold_users_suffix = "allow_cold_users"
        else:
            allow_cold_users_suffix = "only_warm_users"

        if self.user_wise:
            user_wise_string = "user_wise"
        else:
            user_wise_string = "global_sample"


        name_suffix = "_{}_{}".format(allow_cold_users_suffix, user_wise_string)


        dataIO = DataIO(folder_path = save_folder_path)

        split_parameters_dict = dataIO.load_data(file_name ="split_parameters" + name_suffix)

        for attrib_name in split_parameters_dict.keys():
             self.__setattr__(attrib_name, split_parameters_dict[attrib_name])


        self.SPLIT_GLOBAL_MAPPER_DICT = dataIO.load_data(file_name ="split_mappers" + name_suffix)

        self.FOLD_URM_LIST = dataIO.load_data(file_name ="fold_URM" + name_suffix)

        if len(self.dataReader_object.get_loaded_ICM_names())>0:
            self.SPLIT_ICM_DICT = dataIO.load_data(file_name ="split_ICM" + name_suffix)

            self.SPLIT_ICM_MAPPER_DICT = dataIO.load_data(file_name ="split_ICM_mappers" + name_suffix)

        if len(self.dataReader_object.get_loaded_UCM_names())>0:
            self.SPLIT_UCM_DICT = dataIO.load_data(file_name ="split_UCM" + name_suffix)

            self.SPLIT_UCM_MAPPER_DICT = dataIO.load_data(file_name ="split_UCM_mappers" + name_suffix)




    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                DATA CONSISTENCY                                     ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _verify_data_consistency(self):

        self._assert_is_initialized()











    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                ITERATOR                                             ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def __iter__(self):

        self._assert_is_initialized()

        self.__iterator_current_fold = 0
        return self


    def __next__(self):

        fold_to_return = self.__iterator_current_fold

        if self.__iterator_current_fold >= self.n_folds:
            raise StopIteration

        self.__iterator_current_fold += 1

        return self[fold_to_return]


    def __getitem__(self, n_fold):
        """
        :param index:
        :return:
        """

        self._assert_is_initialized()

        return self.FOLD_URM_LIST[str(n_fold)]
