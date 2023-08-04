#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import zipfile, shutil
from recsys.Data_manager.DataReader import DataReader
from recsys.Data_manager.DataReader_utils import download_from_URL
from recsys.Data_manager.DatasetMapperManager import DatasetMapperManager
from recsys.Data_manager.Movielens._utils_movielens_parser import _loadURM, _loadICM_genres_years


def _loadICM_tags(ICM_tags_path):
    ICM_tags_dataframe = pd.read_csv(filepath_or_buffer=ICM_tags_path, header=0, dtype={0:str, 1:str, 2:str, 3:int})
    ICM_tags_dataframe.columns = ["UserID", "ItemID", "FeatureID", "Timestamp"]
    ICM_tags_dataframe = ICM_tags_dataframe.drop(columns=["UserID", "Timestamp"])
    ICM_tags_dataframe.drop_duplicates(subset=["ItemID", "FeatureID"], keep='first', inplace= True)
    ICM_tags_dataframe["Data"] = 1
    return ICM_tags_dataframe


class MovielensLatestSmallReader(DataReader):

    DATASET_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    DATASET_SUBFOLDER = "MovielensLatestSmall/"
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = ["ICM_genres", "ICM_year", "ICM_all", "ICM_tags", "ICM_one_hot"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original
        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-latest-small.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "ml-latest-small.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-latest-small.zip")


        ICM_genre_path = dataFile.extract("ml-latest-small/movies.csv", path=zipFile_path + "decompressed/")
        ICM_tags_path = dataFile.extract("ml-latest-small/tags.csv", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ml-latest-small/ratings.csv", path=zipFile_path + "decompressed/")

        self._print("Loading Interactions")
        URM_all_dataframe, URM_timestamp_dataframe = _loadURM(URM_path, header=0, separator=',')

        self._print("Loading Item Features genres")
        ICM_genres_dataframe, ICM_years_dataframe = _loadICM_genres_years(ICM_genre_path, header=None, separator=',', genresSeparator="|")

        self._print("Loading Item Features Tags")
        ICM_tags_dataframe = _loadICM_tags(ICM_tags_path)

        ICM_one_hot_dataframe = pd.concat([ICM_genres_dataframe, ICM_tags_dataframe])
        ICM_all_dataframe = pd.concat([ICM_one_hot_dataframe, ICM_years_dataframe])

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_timestamp_dataframe, "URM_timestamp")
        dataset_manager.add_ICM(ICM_genres_dataframe, "ICM_genres")
        dataset_manager.add_ICM(ICM_years_dataframe, "ICM_year")
        dataset_manager.add_ICM(ICM_tags_dataframe, "ICM_tags")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")
        dataset_manager.add_ICM(ICM_one_hot_dataframe, "ICM_one_hot")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset
