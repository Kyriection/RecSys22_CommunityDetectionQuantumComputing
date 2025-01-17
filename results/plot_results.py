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

from utils.plot import plot_line, plot_scatter, plot_line_xticks
from utils.urm import get_community_urm, load_data, merge_sparse_matrices, load_data_k_fold
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, \
    MovielensSampleReader, MovielensSample2Reader
from CommunityDetection import BaseCommunityDetection, QUBOBipartiteCommunityDetection, \
    QUBOBipartiteProjectedCommunityDetection, Communities, CommunityDetectionRecommender, \
    get_community_folder_path, KmeansCommunityDetection, HierarchicalClustering, \
    QUBOGraphCommunityDetection, QUBOProjectedCommunityDetection, UserCommunityDetection, \
    HybridCommunityDetection, MultiHybridCommunityDetection, QUBONcutCommunityDetection, \
    QUBOBipartiteProjectedItemCommunityDetection, CommunitiesEI, Clusters, \
    LTBipartiteProjectedCommunityDetection, LTBipartiteCommunityDetection


logging.basicConfig(level=logging.INFO)
QUBO = ["Hybrid", "QUBOBipartiteCommunityDetection", "QUBOBipartiteProjectedCommunityDetection", "KmeansCommunityDetection", "HierarchicalClustering", "UserCommunityDetection"]
# METHOD = ["LeapHybridSampler", "SimulatedAnnealingSampler", "SteepestDescentSolver", "TabuSampler", ""]
METHOD = ["SimulatedAnnealingSampler", ""]
RESULT = ".result_df.csv"
TOTAL_DATA = {'C': {}, 'MAE': {}, 'RMSE': {}, 'W-MAE': {}, 'W-RMSE': {}}
TAIL_DATA = {}
MIN_RATING_NUM = 1
PLOT_CUT = 30
DATA = {key : {'x': None, 'MAE': {}, 'RMSE': {}, 'W-MAE': {}, 'W-RMSE': {}} for key in ['rank', 'rating', 'cluster', 'Number of tail items']}
CT: float = 0.0
STEP = 10

def init_global_data():
  global TOTAL_DATA, DATA, TAIL_DATA
  TOTAL_DATA = {'C': {}, 'MAE': {}, 'RMSE': {}, 'W-MAE': {}, 'W-RMSE': {}}
  DATA = {key : {'x': None, 'MAE': {}, 'RMSE': {}, 'W-MAE': {}, 'W-RMSE': {}} for key in ['rank', 'rating', 'cluster', 'Number of tail items']}
  TAIL_DATA = {}


def process_total_data():
  if len(TOTAL_DATA['C']) == 0: return None, None
  print(TOTAL_DATA)
  # if len(TOTAL_DATA['C']) == 0 or\
    # (len(TOTAL_DATA['C']) == 1 and list(TOTAL_DATA['C'].values())[0] == 1):
    # return None, None
  for key in ['MAE', 'RMSE', 'W-MAE', 'W-RMSE']:
    values = TOTAL_DATA[key].values()
    TOTAL_DATA[key][-2] = min(values)
    # if values:
    #   TOTAL_DATA[key][-2] = min(values)
    # else:
    #    return None, None
  df_total = pd.DataFrame(TOTAL_DATA)
  df_tail = pd.DataFrame(TAIL_DATA)
  df_tail.sort_index(inplace=True)
  return df_total, df_tail


def plot(output_folder_path, show: bool = False):
    df_total, df_tail = process_total_data()
    if df_total is None:
       print('data empty.')
       return
    df_total.to_csv(os.path.join(output_folder_path, f'total_MAE_RMSE.csv'))
    # df_tail.to_csv(os.path.join(output_folder_path, f'tail_MAE_RMSE.csv'))
    # logging.info(f'save results in {output_folder_path}')
    print(f'save results in {output_folder_path}')
    if show:
      print(df_total)
    for key in DATA:
        data = DATA[key]
        x = data['x']
        if x is None:
           continue
        MAE_data = data['MAE']
        RMSE_data = data['RMSE']
        WMAE_data = data['W-MAE']
        WRMSE_data = data['W-RMSE']
        if key in ['rank', 'rating']:
          # plot_scatter(x, MAE_data, output_folder_path, key, 'MAE')
          # plot_scatter(x, RMSE_data, output_folder_path, key, 'RMSE')
          pass
        elif key in ['cluster']:
          plot_line_xticks(x, MAE_data, output_folder_path, key, 'MAE')
          plot_line_xticks(x, RMSE_data, output_folder_path, key, 'RMSE')
        elif key in ['Number of tail items']:
          plot_line(x, MAE_data, output_folder_path, key, 'MAE')
          plot_line(x, RMSE_data, output_folder_path, key, 'RMSE')
          plot_line(x, WMAE_data, output_folder_path, key, 'W-MAE')
          plot_line(x, WRMSE_data, output_folder_path, key, 'W-RMSE')


def concatenate_df(C_quantity, result_df, result_df_ei = None):
  if result_df_ei is None:
    return result_df

  cut_quantity = sorted(C_quantity, reverse=True)[int(len(C_quantity) * CT)]
  head_user_mask = C_quantity > cut_quantity
  tail_user_mask = ~head_user_mask
  # logging.info(f'collect_data: cut at {cut_quantity}, head size: {np.sum(head_user_mask)}, tail size: {np.sum(tail_user_mask)}')
  print(f'collect_data: cut at {cut_quantity}, head size: {np.sum(head_user_mask)}, tail size: {np.sum(tail_user_mask)}')
  data = np.zeros((C_quantity.size, 3)) # [MAE, MSE, num_rating]
  data[head_user_mask] = result_df_ei.values
  data[tail_user_mask] = result_df.values 
  return pd.DataFrame(data, columns=['MAE', 'MSE', 'num_rating'])


def collect_data(C_quantity, n_iter, result_df, result_df_ei = None):
    global MIN_RATING_NUM, PLOT_CUT, CT, STEP
    data = concatenate_df(C_quantity, result_df, result_df_ei).values
    # if CT > 0.0:
    #   cut_quantity = sorted(C_quantity, reverse=True)[int(len(C_quantity) * CT)]
    #   head_user_mask = C_quantity > cut_quantity
    #   tail_user_mask = ~head_user_mask
    #   print(f'collect_data: cut at {cut_quantity}, head size: {np.sum(head_user_mask)}, tail size: {np.sum(tail_user_mask)}')
    #   data = np.zeros((C_quantity.size, 3)) # [MAE, MSE, num_rating]
    #   if result_df_ei is not None:
    #     data[head_user_mask] = result_df_ei.values
    #   data[tail_user_mask] = result_df.values 
    # else:
    #   data: np.ndarray = result_df.values

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
    # calc total error rate at tail 10%, 20% ... 100%
    TAIL_DATA[n_iter] = {}
    # cluster by C_quantity
    tot_mae = 0.0
    tot_mse = 0.0
    tot_num_rating = 0
    cluster_data = {}
    x = []
    MAE_data = []
    RMSE_data = []
    tot_wmae = 0.0
    tot_wmse = 0.0
    WMAE_data = []
    WRMSE_data = []
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
        tot_mse += mse * num_rating
        tot_num_rating += num_rating
        cluster_data[quantity] = _data
        tot_wmae += mae
        tot_wmse += mse

        if tot_num_rating == 0:
           continue
        percent = i / n_users
        p = int(percent * 10)
        mae = round(tot_mae / tot_num_rating, 4)
        rmse = round(np.sqrt(tot_mse / tot_num_rating), 4)
        TAIL_DATA[n_iter][f'MAE-{p * 10 + 10}%'] = mae
        TAIL_DATA[n_iter][f'RMSE-{p * 10 + 10}%'] = rmse

        if i % STEP != (n_users - 1) % STEP:
           continue
        x.append(i)
        MAE_data.append(round(tot_mae / tot_num_rating, 4))
        RMSE_data.append(round(np.sqrt(tot_mse / tot_num_rating), 4))
        WMAE_data.append(round(tot_wmae / (i + 1), 4))
        WRMSE_data.append(round(np.sqrt(tot_wmse / (i + 1)), 4))
    # accumulate error rate from tail to head
    # print(f'n_iter={n_iter}, len(x)={len(x)}, len(MAE_data)={len(MAE_data)}')
    DATA['Number of tail items']['x'] = x
    DATA['Number of tail items']['MAE'][n_iter] = MAE_data
    DATA['Number of tail items']['RMSE'][n_iter] = RMSE_data
    DATA['Number of tail items']['W-MAE'][n_iter] = WMAE_data
    DATA['Number of tail items']['W-RMSE'][n_iter] = WRMSE_data
    TOTAL_DATA['MAE'][n_iter] = MAE_data[-1]
    TOTAL_DATA['RMSE'][n_iter] = RMSE_data[-1]
    TOTAL_DATA['W-MAE'][n_iter] = WMAE_data[-1]
    TOTAL_DATA['W-RMSE'][n_iter] = WRMSE_data[-1]
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
        if num_rating == 0:
           num_rating = 1
        MAE_data.append(MAE / num_rating)
        RMSE_data.append(np.sqrt(MSE / num_rating))
    DATA['cluster']['x'] = x
    DATA['cluster']['MAE'][n_iter] = MAE_data
    DATA['cluster']['RMSE'][n_iter] = RMSE_data


def extract_file(file, tmp):
  try:
    z = zipfile.ZipFile(file, 'r') 
    file_path = z.extract(RESULT, path=tmp)
    result_df = pd.read_csv(file_path, index_col="user")
    return result_df
  except FileNotFoundError as e:
    print(e)
    return None


def get_C(tmp):
  C = 0
  for file in os.listdir(tmp):
    if file[0] == 'C':
      C = int(file[1:])
      break
    # if os.path.isdir(os.path.join(tmp, c)) and c[0] == 'c':
    if 'run' in file or 'model' in file:
      try:
        C = max(C, int(file[-6:-4]) + 1) # run/model{02d}.zip
      except:
        continue
  return C


def print_result_(C_quantity, cut_ratio, data_reader_class, method_list, sampler_list,
                  recommender: str = 'LRRecommender', show: bool = False, output_folder: str = None,
                  output_tag: str = None, result_folder_path: str = './results/'):
  global CT
  CT = cut_ratio
  dataset = data_reader_class.DATASET_SUBFOLDER
  dataset = os.path.abspath(result_folder_path + dataset)
  sampler_list = [sampler.__class__.__name__ for sampler in sampler_list]\
               + [f'{sampler.__class__.__name__}_DWaveSampler' for sampler in sampler_list]
  # special for baseline
  path = os.path.join(dataset, recommender)
  file = os.path.join(path, 'baseline.zip')
  result_df_total = extract_file(file, path)
  if result_df_total is not None:
    init_global_data()
    TOTAL_DATA['C'][-1] = 0
    collect_data(C_quantity, -1, result_df_total)
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

    if 'Kmeans' in method.name or method.name in ['EachItem']:
      sampler_name_list = ['']
    else:
      sampler_name_list = sampler_list
    for m in sampler_name_list:
      print(f'collecting: {path}/iter/{m}')
      # if show:
        # print(m)
      init_global_data()
      result_df_ei = None
      if CT > 0.0:
        name = 'iter-1'
        d = os.path.join(path, name)
        cur = os.path.join(path, d)
        tmp = os.path.join(cur, m)
        file = os.path.join(tmp, f'cd_{recommender}.zip')
        result_df_ei = extract_file(file, cur)
      for name in dir_file:
        d = os.path.join(path, name)
        if len(name) < 4 or name[:4] != 'iter' or not os.path.isdir(d):
          continue
        # print(d)
        cur = os.path.join(path, d)
        # print(cur)
        tmp = os.path.join(cur, m)
        if not os.path.exists(tmp):
          continue
        N = int(name[4:]) + 1
        #print(f'---------N={N}------------')
        if CT > 0.0 and N == 0:
           continue
        C = get_C(tmp)
        file = os.path.join(tmp, f'cd_{recommender}.zip')
        result_df = extract_file(file, tmp)
        if result_df is None:
           continue
        # logging.info(f'extract_file({file}), resutl_df is None: {result_df is None}')
        TOTAL_DATA['C'][N] = C
        collect_data(C_quantity, N, result_df, result_df_ei)

      if len(TOTAL_DATA['C']) == 0: continue
      TOTAL_DATA['C'][0] = 1
      collect_data(C_quantity, 0, result_df_total)
      if output_folder is None:
        output_path = os.path.join(path, m, output_tag)
      else:
        output_path = os.path.join(output_folder, method.name, m, output_tag)
      if not os.path.exists(output_path):
        os.system(f'mkdir -p {output_path}')
      plot(output_path, show)


def print_result(cut_ratio, data_reader_class, method_list, sampler_list, recommender: str = 'LRRecommender',
                 show: bool = False, output_folder: str = None, output_tag: str = None, 
                 result_folder_path: str = './results/'):
  '''
  TODO: load_data() arguments: user_wise, make_implicit, threshold
  '''
  global CT
  CT = cut_ratio
  data_reader = data_reader_class()
  urm_train, urm_validation, urm_test = load_data(data_reader, [80, 10, 10], False, False)
  urm_train, urm_validation, urm_test = urm_train.T.tocsr(), urm_validation.T.tocsr(), urm_test.T.tocsr()
  urm_train_last_test = merge_sparse_matrices(urm_train, urm_validation)
  urm_all = merge_sparse_matrices(urm_train_last_test, urm_test)
  C_quantity = np.ediff1d(urm_all.tocsr().indptr) # count of each row
  print_result_(C_quantity, cut_ratio, data_reader_class, method_list, sampler_list, recommender,
                show, output_folder, output_tag, result_folder_path)


def create_empty_df(n_users):
  return pd.DataFrame(data=np.zeros(shape=(n_users, 3)), columns=['MAE', 'MSE', 'num_rating'], index=range(n_users))


def add_df(df1: pd.DataFrame, df2: pd.DataFrame):
  df1['MAE'] = df1['MAE'] * df1['num_rating'] + df2['MAE'] * df2['num_rating']
  df1['MSE'] = df1['MSE'] * df1['num_rating'] + df2['MSE'] * df2['num_rating']
  df1['num_rating'] += df2['num_rating']
  num_rating = df1['num_rating'].copy()
  num_rating[num_rating == 0.0] = 1.0
  df1['MAE'] /= num_rating
  df1['MSE'] /= num_rating
  return df1


def print_result_k_fold(C_quantity, cut_ratio, data_reader_class, method_list, sampler_list,
                        recommender: str = 'LRRecommender', show: bool = False, output_folder: str = None,
                        output_tag: str = None, result_folder_path: str = './results/', n_folds: int = 5):
  global CT
  CT = cut_ratio
  n_users = len(C_quantity)
  dataset = data_reader_class.DATASET_SUBFOLDER
  sampler_list = [sampler.__class__.__name__ for sampler in sampler_list]\
               + [f'{sampler.__class__.__name__}_DWaveSampler' for sampler in sampler_list]
  
  result_df_total = create_empty_df(n_users)
  for k in range(n_folds):
    path = os.path.join(result_folder_path, f'fold-{k:02d}', dataset, recommender)
    file_path = os.path.join(path, 'baseline.zip')
    df_total = extract_file(file_path, path)
    result_df_total = add_df(result_df_total, df_total)
  init_global_data()
  TOTAL_DATA['C'][-1] = 0
  collect_data(C_quantity, -1, result_df_total)
  plot(path, show)

  for method in method_list:
    if 'Kmeans' in method.name or method.name in ['EachItem']:
      sampler_name_list = ['']
    else:
      sampler_name_list = sampler_list

    for sampler in sampler_name_list:
      results_df = {}
      C_list = {}
      cnt = {}
      for k in range(n_folds):
        path = os.path.join(result_folder_path, f'fold-{k:02d}', dataset, method.name)
        if not os.path.exists(path): break
        dir_files = sorted(os.listdir(path))
        result_df_ei = None
        if CT > 0.0:
          result_path = os.path.join(path, 'iter-1', sampler)
          file_path = os.path.join(result_path, f'cd_{recommender}.zip')
          result_df_ei = extract_file(file_path, result_path)
        for name in dir_files:
          if len(name) < 4 or name[:4] != 'iter': continue
          N = int(name[4:]) + 1
          if CT > 0.0 and N == 0: continue
          result_path = os.path.join(path, name, sampler)
          file_path = os.path.join(result_path, f'cd_{recommender}.zip')
          result_df = extract_file(file_path, result_path)
          if result_df is None: continue
          if result_df_ei is not None:
            result_df = concatenate_df(C_quantity, result_df, result_df_ei)
          C = get_C(result_path)
          results_df[N] = add_df(results_df.get(N, create_empty_df(n_users)), result_df)
          C_list[N] = C_list.get(N, 0) + C
          cnt[N] = cnt.get(N, 0) + 1

      if not results_df: continue
      output_path = os.path.join('./results/', dataset, method.name, sampler, output_tag)
      os.makedirs(output_path, exist_ok=True)
      init_global_data()
      TOTAL_DATA['C'][0] = 1
      collect_data(C_quantity, 0, result_df_total)
      for N in cnt:
        if cnt[N] < n_folds: continue
        TOTAL_DATA['C'][N] = round(C_list[N] / n_folds, 1)
        collect_data(C_quantity, N, results_df[N], None)
        results_df[N].to_csv(os.path.join(output_path, f'results-iter{N:02d}.csv'))
      plot(output_path, show)


def print_result_k_fold_mean(data_reader_class, method_list, sampler_list, recommender: str = 'LRRecommender',
                             output_folder: str = None, output_tag: str = None, results_folder_path: str = './results/',
                             n_folds: int = 5):
  print('---------print_result_k_fold--------------')
  dataset = data_reader_class.DATASET_SUBFOLDER
  sampler_list = [sampler.__class__.__name__ for sampler in sampler_list]\
               + [f'{sampler.__class__.__name__}_DWaveSampler' for sampler in sampler_list]
  for method in method_list:
    if 'Kmeans' in method.name or method.name in ['EachItem']:
      sampler_name_list = ['']
    else:
      sampler_name_list = sampler_list
    for m in sampler_name_list:
      df_list = [None] * n_folds
      df_flag = True
      for k in range(n_folds):
        result_folder_path = f'{results_folder_path}fold-{k:02d}/'
        dataset_path = os.path.abspath(result_folder_path + dataset)
        path = os.path.join(dataset_path, method.name)
        if output_folder is None:
          output_path = os.path.join(path, m, output_tag)
        else:
          output_path = os.path.join(output_folder, method.name, m, output_tag)
        try:
          df_list[k] = pd.read_csv(os.path.join(output_path, f'total_MAE_RMSE.csv'))
        except Exception as e:
          df_flag = False
          print(e)
          break
      if not df_flag:
        print(f'{dataset}/{method.name}/{m}: fail to read n_folds results.')
        continue

      df_sum = df_list[0]
      df_min = df_list[0]
      df_max = df_list[0]
      for i in range(1, n_folds):
        df_sum = df_sum + df_list[i]
        for index in df_sum.index:
          for column in df_sum.columns:
            df_min.loc[index, column] = min(df_min.loc[index, column], df_list[i].loc[index, column])
            df_max.loc[index, column] = max(df_max.loc[index, column], df_list[i].loc[index, column])
      df_mean = df_sum / n_folds
      for index in df_mean.index:
        for column in df_mean.columns:
          val_min = df_min.loc[index, column]
          val_max = df_max.loc[index, column]
          val_mean = df_mean.loc[index, column]
          val_error = max(val_mean - val_min, val_max - val_mean)
          df_mean.loc[index, column] = f'{val_mean:.4f}±{val_error:.4f}'
      
      result_folder_path = './results/'
      dataset_path = os.path.abspath(result_folder_path + dataset)
      path = os.path.join(dataset_path, method.name)
      output_path = os.path.join(path, m, output_tag)
      os.makedirs(output_path, exist_ok=True)
      print(f'save df_mean at {output_path}')
      df_mean.to_csv(os.path.join(output_path, f'{n_folds}-fold-results.csv'))


if __name__ == '__main__':
  # dataset = input("input file folder name: ")
  # cut_ratio = float(input('input CT cut ration: '))
  # show = input("print on CMD or not: ")
  # show = True if show else False
  show = True
  # print_result(cut_ratio, MovielensSample2Reader, [QUBOLongTailCommunityDetection], show)
  # print_result(cut_ratio, MovielensSample2Reader, [KmeansCommunityDetection], show)
  # print_result(cut_ratio, Movielens100KReader, [QUBOBipartiteCommunityDetection], show)
  result_folder_path = 'results/'
  output_folder = os.path.join('./results/', 'Movielens100K', 'results')
  # print_result(0, Movielens100KReader, [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection], False, output_folder, '5_1.0_0.0_0', result_folder_path)
  print_result(0, Movielens100KReader, [LTBipartiteProjectedCommunityDetection], False, output_folder, '3_0.005_0.0_0', '/app/results-LTBipartiteProjectedCommunityDetection-3-0.005/')