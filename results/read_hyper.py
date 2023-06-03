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
  folder_path = os.path.abspath(folder_path)
  for hyper_name in os.listdir(folder_path):
    hyper_path = os.path.join(folder_path, hyper_name)
    if not os.path.isdir(hyper_path):
      continue
    hyper_file = os.path.join(hyper_path, 'total_MAE_RMSE.csv')
    pd_reader = pd.read_csv(hyper_file)
    mae = min(pd_reader['MAE'])
    rmse = min(pd_reader['RMSE'])
    hypers = [float(hyper) for hyper in hyper_name.split('_')]
    T, alpha, cut_ratio = hypers
    MAE[T] = MAE.get(T, {})
    RMSE[T] = RMSE.get(T, {})
    # MAE[T][alpha] = mae
    MAE[T][cut_ratio] = mae
    # RMSE[T][alpha] = rmse
    RMSE[T][cut_ratio] = rmse
  df_mae = pd.DataFrame(MAE)
  df_rmse = pd.DataFrame(RMSE)
  df_mae.sort_index(inplace=True)
  df_rmse.sort_index(inplace=True)
  df_mae.to_csv(os.path.join(folder_path, 'MAE.csv'))
  df_rmse.to_csv(os.path.join(folder_path, 'RMSE.csv'))

if __name__ == '__main__':
  folder_path = input('input results folder path: ')
  read_results(folder_path)
  