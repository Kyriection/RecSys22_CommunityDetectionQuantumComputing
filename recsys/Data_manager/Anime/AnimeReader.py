import json, gzip
import pandas as pd
from recsys.Data_manager.DataReader import DataReader
from recsys.Data_manager.DataReader_utils import download_from_URL
from recsys.Data_manager.DatasetMapperManager import DatasetMapperManager


def loadURM(json_file, user_dict, item_dict):
    '''
user_id - non identifiable randomly generated user id.
anime_id - the anime that this user has rated.
rating - rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating).
    '''
    URM_list = []
    for l in json_file:
        data = eval(l)
        user = data['reviewerID']
        item = data['asin']
        if user not in user_dict:
            user_dict[user] = str(len(user_dict) + 1)
        if item not in item_dict:
            item_dict[item] = str(len(item_dict) + 1)
        URM_list.append([user_dict[user], item_dict[item], data['overall']])

    URM_dataframe = pd.DataFrame(URM_list, columns=["UserID", "ItemID", "Data"])
    return URM_dataframe


def loadICM(json_file, item_dict):
    '''
anime_id - myanimelist.net's unique id identifying an anime.
name - full name of anime.
genre - comma separated list of genres for this anime.
type - movie, TV, OVA, etc.
episodes - how many episodes in this show. (1 if movie).
rating - average rating out of 10 for this anime.
members - number of community members that are in this anime's "group".
    '''
    ICM_quantization_list = []
    ICM_one_hot_list = []
    for l in json_file:
        data = eval(l)
        item = data['asin']
        if item not in item_dict: continue

        if 'price' in data:
            ICM_quantization_list.append(['price', item_dict[item], data['price']])
        if 'salesRank' in data:
            ICM_quantization_list.append(['salesRank', item_dict[item], data['salesRank']])
        if 'brand' in data:
            ICM_one_hot_list.append([data['brand'], item_dict[item], 1])
        for category in data['categories']:
            category_name = '_'.join(category)
            ICM_one_hot_list.append([category_name, item_dict[item], 1])

    ICM_all_list = ICM_quantization_list + ICM_one_hot_list
    ICM_all_dataframe = pd.DataFrame(ICM_all_list, columns=['FeatureID', 'ItemID', 'Data'])
    ICM_one_hot_dataframe = pd.DataFrame(ICM_one_hot_list, columns=['FeatureID', 'ItemID', 'Data'])
    return ICM_all_dataframe, ICM_one_hot_dataframe


class AnimeReader(DataReader):
    # https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
    # https://github.com/Mayank-Bhatia/Anime-Recommender/tree/master/data

    # DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    # DATASET_SUBFOLDER = "Amazon/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_all", "ICM_one_hot"]
    # AVAILABLE_UCM = ["UCM_all"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original
        pass

        # zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        # try:
        #     meta_file = gzip.open(zipFile_path + "meta.json.gz", 'r')
        #     reviews_file = gzip.open(zipFile_path + "reviews.json.gz", 'r')
        # except (FileNotFoundError):
        #     self._print("Unable to find data zip file. Downloading...")

        #     download_from_URL(self.URL_METADATA, zipFile_path, "meta.json.gz")
        #     download_from_URL(self.URL_REVIEWS, zipFile_path, "reviews.json.gz")

        #     meta_file = gzip.open(zipFile_path + "meta.json.gz", 'r')
        #     reviews_file = gzip.open(zipFile_path + "reviews.json.gz", 'r')

        # user_dict = {}
        # item_dict = {}

        # self._print("Loading Interactions")
        # URM_dataframe = loadURM(reviews_file, user_dict, item_dict)
        # reviews_file.close()

        # self._print("Loading Item Features genres")
        # ICM_all_dataframe, ICM_one_hot_dataframe = loadICM(meta_file, item_dict)
        # meta_file.close()

        # dataset_manager = DatasetMapperManager()
        # dataset_manager.add_URM(URM_dataframe, "URM_all")
        # dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")
        # dataset_manager.add_ICM(ICM_one_hot_dataframe, "ICM_one_hot")

        # loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
        #                                                   is_implicit=self.IS_IMPLICIT)

        # self._print("Loading Complete")
        # return loaded_dataset