#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""


import pandas as pd
import zipfile, shutil
from recsys.Data_manager.DataReader import DataReader
from recsys.Data_manager.DataReader_utils import download_from_URL
from recsys.Data_manager.DatasetMapperManager import DatasetMapperManager
from recsys.Data_manager.Movielens._utils_movielens_parser import _loadURM

def _loadICM_genres_years(genres_path, header=True, separator=',', genresSeparator="|"):

    ICM_genres_dataframe = pd.read_csv(filepath_or_buffer=genres_path, sep=separator, header=header,
                                       dtype={i:str for i in range(24)}, engine='python')
    genres_columns = [f'Genera_{i}' for i in range(19)]
    ICM_genres_dataframe.columns = ["ItemID", "Title", "Date", "", "URL", *genres_columns]

    ICM_years_dataframe = ICM_genres_dataframe.copy()
    ICM_years_dataframe["Year"] = ICM_years_dataframe["Title"].str.extract(pat='\(([0-9]+)\)')
    ICM_years_dataframe = ICM_years_dataframe[ICM_years_dataframe["Year"].notnull()]
    ICM_years_dataframe["Year"] = ICM_years_dataframe["Year"].astype(int)
    ICM_years_dataframe = ICM_years_dataframe[['ItemID', 'Year']]
    ICM_years_dataframe.rename(columns={'Year': 'Data'}, inplace=True)
    ICM_years_dataframe["FeatureID"] = "Year"


    ICM_genre_list = [[genere for genere in genres_columns if ICM_genres_dataframe[genere][index] == '1'] for index in range(len(ICM_genres_dataframe))]
    ICM_genres_dataframe = pd.DataFrame(ICM_genre_list, index=ICM_genres_dataframe["ItemID"]).stack()
    ICM_genres_dataframe = ICM_genres_dataframe.reset_index()[[0, 'ItemID']]
    ICM_genres_dataframe.columns = ['FeatureID', 'ItemID']
    ICM_genres_dataframe["Data"] = 1
    # ICM_genre_list = []
    # for index in range(len(ICM_genres_dataframe)):
    #     for genere in genres_columns:
    #         value = ICM_genres_dataframe[genere][index]
    #         if value == '1':
    #             ICM_genre_list.append([str(index), genere, 1])
    # ICM_genres_dataframe = pd.DataFrame(ICM_genre_list, columns=['ItemID', 'FeatureID', 'Data'])

    return ICM_genres_dataframe, ICM_years_dataframe


class MovielensSample3Reader(DataReader):
    '''
    rating: 951
    user: 61
    item: 108
    '''

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-sample.zip"
    DATASET_SUBFOLDER = "MovielensSample3/"
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = ["ICM_genres", "ICM_year", "ICM_all", "ICM_one_hot"]
    AVAILABLE_UCM = ["UCM_all"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-sample.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "ml-sample.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-sample.zip")


        ICM_genre_path = dataFile.extract("ml-sample/u.item", path=zipFile_path + "decompressed/")
        UCM_path = dataFile.extract("ml-sample/u.user", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ml-sample/u.data", path=zipFile_path + "decompressed/")

        self._print("Loading Interactions")
        URM_all_dataframe, URM_timestamp_dataframe = _loadURM(URM_path, header=None, separator='\t')

        self._print("Loading Item Features genres")
        ICM_genres_dataframe, ICM_years_dataframe = _loadICM_genres_years(ICM_genre_path, header=None, separator='|', genresSeparator="|")
        ICM_all_dataframe = pd.concat([ICM_genres_dataframe, ICM_years_dataframe])

        self._print("Loading User Features")
        UCM_dataframe = pd.read_csv(filepath_or_buffer=UCM_path, sep="|", header=None, dtype={0:str, 1:str, 2:str, 3:str, 4:str}, engine='python')
        UCM_dataframe.columns = ["UserID", "age_group", "gender", "occupation", "zip_code"]

        # For each user a list of features
        # UCM_dataframe = occupation_to_number(UCM_dataframe)
        # UCM_list = [[feature_name + "_" + str(UCM_dataframe[feature_name][index]) for feature_name in ["gender", "age_group", "occupation", "zip_code"]] for index in range(len(UCM_dataframe))]
        # UCM_dataframe = pd.DataFrame(UCM_list, index=UCM_dataframe["UserID"]).stack()
        # UCM_dataframe = UCM_dataframe.reset_index()[[0, 'UserID']]
        # UCM_dataframe.columns = ['FeatureID', 'UserID']
        # UCM_dataframe["Data"] = 1
        UCM_list = [[feature_name, str(index), UCM_dataframe[feature_name][index]] for feature_name in ["age_group"] for index in range(len(UCM_dataframe))] # num
        UCM_list += [[f'{feature_name}_{UCM_dataframe[feature_name][index]}', str(index), 1] for feature_name in ["gender", "occupation"] for index in range(len(UCM_dataframe))] # one-hot
        UCM_dataframe = pd.DataFrame(UCM_list, columns=['FeatureID', 'UserID', 'Data'])


        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_timestamp_dataframe, "URM_timestamp")
        dataset_manager.add_ICM(ICM_genres_dataframe, "ICM_genres")
        dataset_manager.add_ICM(ICM_years_dataframe, "ICM_year")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")
        dataset_manager.add_ICM(ICM_genres_dataframe, "ICM_one_hot")
        dataset_manager.add_UCM(UCM_dataframe, "UCM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset
