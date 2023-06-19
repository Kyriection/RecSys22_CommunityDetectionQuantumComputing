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

logging.basicConfig(level=logging.INFO)

def read_results(folder_path):
  MAE = {}
  RMSE = {}
  WMAE = {}
  WRMSE = {}
  folder_path = os.path.abspath(folder_path)
  for hyper_name in os.listdir(folder_path):
    hyper_path = os.path.join(folder_path, hyper_name)
    if not os.path.isdir(hyper_path):
      continue
    hyper_file = os.path.join(hyper_path, 'total_MAE_RMSE.csv')
    pd_reader = pd.read_csv(hyper_file)
    mae = min(pd_reader['MAE'])
    rmse = min(pd_reader['RMSE'])
    wmae = min(pd_reader['W-MAE'])
    wrmse = min(pd_reader['W-RMSE'])
    hypers = [float(hyper) for hyper in hyper_name.split('_')]
    T, alpha, cut_ratio, layer = hypers
    MAE[T] = MAE.get(T, {})
    RMSE[T] = RMSE.get(T, {})
    WMAE[T] = WMAE.get(T, {})
    WRMSE[T] = WRMSE.get(T, {})
    MAE[T][alpha] = mae
    # MAE[T][cut_ratio] = mae
    RMSE[T][alpha] = rmse
    # RMSE[T][cut_ratio] = rmse
    WMAE[T][alpha] = wmae
    WRMSE[T][alpha] = wrmse
  df_mae = pd.DataFrame(MAE)
  df_rmse = pd.DataFrame(RMSE)
  df_wmae = pd.DataFrame(WMAE)
  df_wrmse = pd.DataFrame(WRMSE)
  df_mae.sort_index(inplace=True)
  df_rmse.sort_index(inplace=True)
  df_wmae.sort_index(inplace=True)
  df_wrmse.sort_index(inplace=True)
  df_mae.to_csv(os.path.join(folder_path, 'MAE.csv'))
  df_rmse.to_csv(os.path.join(folder_path, 'RMSE.csv'))
  df_wmae.to_csv(os.path.join(folder_path, 'W-MAE.csv'))
  df_wrmse.to_csv(os.path.join(folder_path, 'W-RMSE.csv'))

if __name__ == '__main__':
  folder_path = input('input results folder path: ')
  read_results(folder_path)
  