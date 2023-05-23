'''
Author: Kaizyn
Date: 2023-01-21 10:19:24
LastEditTime: 2023-01-21 10:55:37
'''
import os, zipfile, sys, logging

import pandas as pd
import numpy as np

p = os.path.abspath('.')
sys.path.insert(1, p)

from utils.plot import plot_line, plot_scatter
from utils.urm import get_community_urm, load_data, merge_sparse_matrices
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, \
    MovielensSampleReader, MovielensSample2Reader
from CommunityDetection import BaseCommunityDetection, QUBOBipartiteCommunityDetection, \
    QUBOBipartiteProjectedCommunityDetection, Communities, CommunityDetectionRecommender, \
    get_community_folder_path, KmeansCommunityDetection, HierarchicalClustering, \
    QUBOGraphCommunityDetection, QUBOProjectedCommunityDetection, UserCommunityDetection, \
    HybridCommunityDetection, MultiHybridCommunityDetection, QUBONcutCommunityDetection, \
    QUBOBipartiteProjectedItemCommunityDetection, CommunitiesEI, \
    TMPCD, QUBOLongTailCommunityDetection, Clusters


logging.basicConfig(level=logging.INFO)
QUBO = ["Hybrid", "QUBOBipartiteCommunityDetection", "QUBOBipartiteProjectedCommunityDetection", "KmeansCommunityDetection", "HierarchicalClustering", "UserCommunityDetection"]
# METHOD = ["LeapHybridSampler", "SimulatedAnnealingSampler", "SteepestDescentSolver", "TabuSampler", ""]
METHOD = ["SimulatedAnnealingSampler", ""]
CDR = "cd_LRRecommender.zip"
RECOMMENDER = 'LRRecommender'
RESULT = ".result_df.csv"
TOTAL_DATA = {'C': {}, 'MAE': {}, 'RMSE': {}}
MIN_RATING_NUM = 1
PLOT_CUT = 30
DATA = {key : {'x': None, 'MAE': {}, 'RMSE': {}} for key in ['rank', 'rating', 'cluster']}
CT: float = 0.0

def init_global_data():
  global TOTAL_DATA, DATA
  TOTAL_DATA = {'C': {}, 'MAE': {}, 'RMSE': {}}
  DATA = {key : {'x': None, 'MAE': {}, 'RMSE': {}} for key in ['rank', 'rating', 'cluster']}


def plot(output_folder_path, show: bool = False):
    df = pd.DataFrame(TOTAL_DATA)
    if df.shape[0] < 1:
       print('data empty.')
       return
    output_path = os.path.join(output_folder_path, f'total_MAE_RMSE.csv')
    df.to_csv(output_path)
    print(output_path)
    if show:
      print(df)
    for key in ['rank', 'rating']:
        data = DATA[key]
        x = data['x']
        if x is None:
           continue
        MAE_data = data['MAE']
        RMSE_data = data['RMSE']
        plot_scatter(x, MAE_data, output_folder_path, key, 'MAE')
        plot_scatter(x, RMSE_data, output_folder_path, key, 'RMSE')
    for key in ['cluster']:
        data = DATA[key]
        x = data['x']
        if x is None:
           continue
        MAE_data = data['MAE']
        RMSE_data = data['RMSE']
        plot_line(x, MAE_data, output_folder_path, key, 'MAE')
        plot_line(x, RMSE_data, output_folder_path, key, 'RMSE')


def collect_data(urm, n_iter, result_df, result_df_ei = None):
    global MIN_RATING_NUM, PLOT_CUT, CT
    C_quantity = np.ediff1d(urm.tocsr().indptr) # count of each row
    # if result_df_ei is not None:
    if CT > 0.0:
      cut_quantity = sorted(C_quantity, reverse=True)[int(len(C_quantity) * CT)]
      head_user_mask = C_quantity > cut_quantity
      tail_user_mask = ~head_user_mask
      data = np.zeros((C_quantity.size, 3)) # [MAE, MSE, num_rating]
      # print(result_df_ei.values.shape, result_df.values.shape)
      if result_df_ei is not None:
        data[head_user_mask] = result_df_ei.values
      data[tail_user_mask] = result_df.values 
    else:
      data: np.ndarray = result_df.values
    # delete users whose test rating num < MIN_RATING_NUM
    ignore_users = data[:, 2] < MIN_RATING_NUM
    data = data[~ignore_users]
    C_quantity = C_quantity[~ignore_users]
    n_users = C_quantity.size
    # sort by train rating num
    data = data[np.argsort(C_quantity)]
    C_quantity = np.sort(C_quantity)

    # plot by item rank
    DATA['rank']['x'] = list(range(n_users))
    DATA['rank']['MAE'][n_iter] = [mae for mae, mse, num_rating in data]
    DATA['rank']['RMSE'][n_iter] = [np.sqrt(mse) for mae, mse, num_rating in data]
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
    x = list(cluster_data.keys())
    MAE_data = []
    RMSE_data = []
    for key in x:
        mae, mse, num_rating = cluster_data[key]
        MAE_data.append(mae / num_rating)
        RMSE_data.append(np.sqrt(mse / num_rating))
    # plot by #ratings
    DATA['rating']['x'] = x
    DATA['rating']['MAE'][n_iter] = MAE_data
    DATA['rating']['RMSE'][n_iter] = RMSE_data
    # print tot
    tot_mae = round(tot_mae / tot_num_rating, 4)
    tot_rmse = round(np.sqrt(tot_rmse / tot_num_rating), 4)
    TOTAL_DATA['MAE'][n_iter] = tot_mae
    TOTAL_DATA['RMSE'][n_iter] = tot_rmse

    # cluster to PLOT_CUT points
    '''
    def get_cut_size(cut_num: int):
        l = 1
        r = urm.size
        while l < r:
            mid = (l + r + 1) // 2
            cnt = 0
            cur = 0
            for key in x:
                cur += cluster_data[key][2]
                if cur >= mid:
                    cur = 0
                    cnt += 1
            if cur > 0:
                cnt += 1
            # logging.info(f'({l}, {r}), mid={mid}, cnt={cnt}')
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
        _data = cluster_data[key]
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
    DATA['cluster']['x'] = x
    DATA['cluster']['MAE'][n_iter] = MAE_data
    DATA['cluster']['RMSE'][n_iter] = RMSE_data


def extract_file(file, cur):
  try:
    z = zipfile.ZipFile(file, 'r') 
    file_path = z.extract(RESULT, path=cur + "/decompressed")
    result_df = pd.read_csv(file_path, index_col="user")
    return result_df
  except FileNotFoundError as e:
    print(e)
    return None

def print_result(cut_ratio, data_reader_class, method_list, show: bool = False, output_folder: str = None):
  global CT, RECOMMENDER
  CT = cut_ratio
  data_reader = data_reader_class()
  urm_train, urm_validation, urm_test = load_data(data_reader, [80, 10, 10], False, False)
  urm_train, urm_validation, urm_test = urm_train.T.tocsr(), urm_validation.T.tocsr(), urm_test.T.tocsr()# item is main charactor
  dataset = data_reader._get_dataset_name()
  dataset = os.path.abspath("/app/results/" + dataset)
  # special for baseline
  path = os.path.join(dataset, RECOMMENDER)
  file = os.path.join(path, 'baseline.zip')
  result_df = extract_file(file, path)
  collect_data(urm_train, -1, result_df)
  plot(path, show)
  # for method in QUBO:
  # for method in os.listdir(dataset):
  for method in method_list:
    path = os.path.join(dataset, method.name)
    if not os.path.exists(path) or os.path.isfile(path):
      continue
    if show:
      print(method.name)
      # print("N", COL)
    # print(path)
    dir_file = os.listdir(path)
    dir_file.sort()
    for m in METHOD:
      # if show:
        # print(m)
      init_global_data()
      result_df_ei = None
      if CT > 0.0:
        name = 'iter-1'
        d = os.path.join(path, name)
        cur = os.path.join(path, d)
        tmp = os.path.join(cur, m)
        file = os.path.join(tmp, CDR)
        result_df_ei = extract_file(file, cur)
      for name in dir_file:
        d = os.path.join(path, name)
        if not os.path.isdir(d):
          continue
        # print(d)
        cur = os.path.join(path, d)
        # print(cur)
        tmp = os.path.join(cur, m)
        if not os.path.exists(tmp):
          continue
        N = int(name[4:]) + 1
        if CT > 0.0 and N == 0:
           continue
        C = 0
        for c in os.listdir(tmp):
          if os.path.isdir(os.path.join(tmp, c)) and c[0] == 'c':
            try:
              C = max(C, int(c[1:]))
            except:
              continue
        file = os.path.join(tmp, CDR)
        result_df = extract_file(file, cur)
        if result_df is None:
           continue
        # logging.info(f'extract_file({file}), resutl_df is None: {result_df is None}')
        TOTAL_DATA['C'][N] = C + 1
        collect_data(urm_train, N, result_df, result_df_ei)

      # df = pd.DataFrame(TOTAL_DATA)
      # if output_folder is None:
      #   output_folder = path
      # output_path = os.path.join(output_folder, f'{method}_{m}.csv')
      # df.to_csv(output_path)
      # if show:
      #   print(df)
      if output_folder is None:
        output_folder = path
      plot(output_folder, show)


if __name__ == '__main__':
  # dataset = input("input file folder name: ")
  cut_ratio = float(input('input CT cut ration: '))
  # show = input("print on CMD or not: ")
  # show = True if show else False
  show = True
  # print_result(cut_ratio, MovielensSample2Reader, [QUBOLongTailCommunityDetection], show)
  print_result(cut_ratio, MovielensSample2Reader, [KmeansCommunityDetection], show)
  