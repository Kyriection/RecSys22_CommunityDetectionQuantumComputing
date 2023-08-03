import json, gzip
import pandas as pd
from recsys.Data_manager.DataReader import DataReader
from recsys.Data_manager.DataReader_utils import download_from_URL
from recsys.Data_manager.DatasetMapperManager import DatasetMapperManager


def loadURM(json_file, user_dict, item_dict):
    '''
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
asin - ID of the product, e.g. 0000013714
reviewerName - name of the reviewer
helpful - helpfulness rating of the review, e.g. 2/3
reviewText - text of the review
overall - rating of the product
summary - summary of the review
unixReviewTime - time of the review (unix time)
reviewTime - time of the review (raw)
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
{
  "asin": "0000031852",
  "title": "Girls Ballet Tutu Zebra Hot Pink",
  "price": 3.17,
  "imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
  "related":
  {
    "also_bought": ["B00JHONN1S", "B002BZX8Z6", "B00D2K1M3O", "0000031909", "B00613WDTQ", "B00D0WDS9A", "B00D0GCI8S", "0000031895", "B003AVKOP2", "B003AVEU6G", "B003IEDM9Q", "B002R0FA24", "B00D23MC6W", "B00D2K0PA0", "B00538F5OK", "B00CEV86I6", "B002R0FABA", "B00D10CLVW", "B003AVNY6I", "B002GZGI4E", "B001T9NUFS", "B002R0F7FE", "B00E1YRI4C", "B008UBQZKU", "B00D103F8U", "B007R2RM8W"],
    "also_viewed": ["B002BZX8Z6", "B00JHONN1S", "B008F0SU0Y", "B00D23MC6W", "B00AFDOPDA", "B00E1YRI4C", "B002GZGI4E", "B003AVKOP2", "B00D9C1WBM", "B00CEV8366", "B00CEUX0D8", "B0079ME3KU", "B00CEUWY8K", "B004FOEEHC", "0000031895", "B00BC4GY9Y", "B003XRKA7A", "B00K18LKX2", "B00EM7KAG6", "B00AMQ17JA", "B00D9C32NI", "B002C3Y6WG", "B00JLL4L5Y", "B003AVNY6I", "B008UBQZKU", "B00D0WDS9A", "B00613WDTQ", "B00538F5OK", "B005C4Y4F6", "B004LHZ1NY", "B00CPHX76U", "B00CEUWUZC", "B00IJVASUE", "B00GOR07RE", "B00J2GTM0W", "B00JHNSNSM", "B003IEDM9Q", "B00CYBU84G", "B008VV8NSQ", "B00CYBULSO", "B00I2UHSZA", "B005F50FXC", "B007LCQI3S", "B00DP68AVW", "B009RXWNSI", "B003AVEU6G", "B00HSOJB9M", "B00EHAGZNA", "B0046W9T8C", "B00E79VW6Q", "B00D10CLVW", "B00B0AVO54", "B00E95LC8Q", "B00GOR92SO", "B007ZN5Y56", "B00AL2569W", "B00B608000", "B008F0SMUC", "B00BFXLZ8M"],
    "bought_together": ["B002BZX8Z6"]
  },
  "salesRank": {"Toys & Games": 211836},
  "brand": "Coxlures",
  "categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
}
asin - ID of the product, e.g. 0000031852
title - name of the product
price - price in US dollars (at time of crawl)
imUrl - url of the product image
related - related products (also bought, also viewed, bought together, buy after viewing)
salesRank - sales rank information
brand - brand name
categories - list of categories the product belongs to
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


class AmazonReader(DataReader):
    # https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
    # TODO: replace reviews with ratings only

    # DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    # URL_METADATA = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Amazon_Instant_Video.json.gz'
    # URL_REVIEWS = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video.json.gz'
    # DATASET_SUBFOLDER = "Amazon/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_all", "ICM_one_hot"]
    # AVAILABLE_UCM = ["UCM_all"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:
            meta_file = gzip.open(zipFile_path + "meta.json.gz", 'r')
            reviews_file = gzip.open(zipFile_path + "reviews.json.gz", 'r')
        except (FileNotFoundError):
            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.URL_METADATA, zipFile_path, "meta.json.gz")
            download_from_URL(self.URL_REVIEWS, zipFile_path, "reviews.json.gz")

            meta_file = gzip.open(zipFile_path + "meta.json.gz", 'r')
            reviews_file = gzip.open(zipFile_path + "reviews.json.gz", 'r')

        user_dict = {}
        item_dict = {}

        self._print("Loading Interactions")
        URM_dataframe = loadURM(reviews_file, user_dict, item_dict)
        reviews_file.close()

        self._print("Loading Item Features genres")
        ICM_all_dataframe, ICM_one_hot_dataframe = loadICM(meta_file, item_dict)
        meta_file.close()

        self._print(f'{len(user_dict)}, {len(item_dict)}')

        # self._print("Loading User Features")

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_dataframe, "URM_all")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")
        dataset_manager.add_ICM(ICM_one_hot_dataframe, "ICM_one_hot")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Loading Complete")
        return loaded_dataset


class AmazonInstantVideoReader(AmazonReader):

    URL_METADATA = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Amazon_Instant_Video.json.gz'
    URL_REVIEWS = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video.json.gz'
    DATASET_SUBFOLDER = "AmazonInstantVideo/"