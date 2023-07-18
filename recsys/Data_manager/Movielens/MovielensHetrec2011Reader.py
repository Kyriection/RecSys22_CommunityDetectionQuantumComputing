#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/02/2019

@author: Anonymous
"""



import zipfile, shutil
import pandas as pd
from recsys.Data_manager.DatasetMapperManager import DatasetMapperManager
from recsys.Data_manager.Dataset import Dataset
from recsys.Data_manager.DataReader import DataReader
from recsys.Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class MovielensHetrec2011Reader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip"
    DATASET_SUBFOLDER = "MovielensHetrec2011/"
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_year", "ICM_one_hot"]
    # AVAILABLE_UCM = ["UCM_all"]

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "hetrec2011-movielens-2k-v2.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "hetrec2011-movielens-2k-v2.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "hetrec2011-movielens-2k-v2.zip")


        URM_path = dataFile.extract("user_ratedmovies.dat", path=zipFile_path + "decompressed/")
        ICM_genre_path = dataFile.extract("movie_genres.dat", path=zipFile_path + "decompressed/")
        ICM_years_path = dataFile.extract("movies.dat", path=zipFile_path + "decompressed/")


        self._print("Loading Interactions")
        URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep="\t", header=0,
                                        dtype={0:str, 1:str, 2:float}, usecols=[0, 1, 2])
        URM_all_dataframe.columns = ["UserID", "ItemID", "Data"]

        ICM_genres_dataframe = pd.read_csv(filepath_or_buffer=ICM_genre_path, sep="\t", header=None, dtype={0:str, 1:str}, engine='python')
        ICM_genres_dataframe.columns = ["ItemID", "genre"]
        ICM_genre_list = [[f'{feature_name}_{ICM_genres_dataframe[feature_name][index]}', str(index), 1] for feature_name in ["genre"] for index in range(len(ICM_genres_dataframe))] # one-hot
        ICM_genres_dataframe = pd.DataFrame(ICM_genre_list, columns=['FeatureID', 'UserID', 'Data'])

        ICM_years_dataframe = pd.read_csv(filepath_or_buffer=ICM_years_path, sep="\t", header=0,
                                        dtype={0:str, 1:str, 2:str, 3:str, 4:str, 5:int}, usecols=[0, 5])
        ICM_years_dataframe.columns = ["ItemID", "Year"]
        ICM_years_dataframe.rename(columns={'Year': 'Data'}, inplace=True)
        ICM_years_dataframe["FeatureID"] = "Year"

        ICM_all_dataframe = pd.concat([ICM_genres_dataframe, ICM_years_dataframe])


        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_ICM(ICM_genres_dataframe, "ICM_genres")
        dataset_manager.add_ICM(ICM_years_dataframe, "ICM_year")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")
        dataset_manager.add_ICM(ICM_genres_dataframe, "ICM_one_hot")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)


        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset

