from typing import List

import matplotlib
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np

from CommunityDetection.Communities import Communities


def percentile(a: List[list], p: int = 99):
  b = [abs(j) for i in a for j in i]
  b.sort()
  idx = len(b) * p // 100
  # for _p in [80, 90, 95, 98, 99]:
    # print(_p, b[len(b) * _p // 100])
  return b[idx]


def plot_pies(ratios_list: list, user_nums_list: list,
              vmin, vmax, label, file_name: str, cmap = None):
  assert len(ratios_list) == len(user_nums_list)
  num_iters = len(ratios_list)
  fig, ax = plt.subplots()
  if cmap is None:
    cmap = matplotlib.cm.get_cmap(name='viridis')
  norm = matplotlib.colors.Normalize(vmin, vmax)
  smap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
  fig.colorbar(smap, label=f'{label} % ratio')
  # for i in range(num_iters - 1, -1, -1):
  for i in range(num_iters):
    ratios = ratios_list[i]
    user_nums = user_nums_list[i]

    pie = ax.pie(
      x = user_nums,
      # autopct = '%3.1f%%',
      radius = (i + 1) / num_iters,
      # pctdistance = 0.85,
      colors = smap.to_rgba(ratios),
      # startangle = 180,
      # textprops = {'color': 'w'},
      wedgeprops = {'width': 1 / num_iters, 'edgecolor': 'w'}
    )
    # plt.savefig(output_path[:-4] + f'_{i}.png')
  plt.savefig(file_name)


def plot_metric(communities: Communities, output_folder: str = '',
         cutoff: int = 10, metric: str = 'NDCG'):
  num_iters = communities.num_iters + 1
  ratios_list = []
  user_nums_list = []
  for i in range(num_iters):
    ratios = []
    user_nums = []
    for community in communities.iter(i):
      value_test = community.result_dict_test['result_df'].loc[cutoff, metric]
      value_baseline = community.result_dict_baseline['result_df'].loc[cutoff, metric]
      if value_baseline > 0:
        ratio = (value_test - value_baseline) / value_baseline
      else:
        ratio = 0.0
      ratios.append(ratio)
      user_nums.append(len(community.users))
      # print(f'value_test={value_test}, value_baseline={value_baseline}, ratio={ratio}.')
    ratios_list.append(ratios)
    user_nums_list.append(user_nums)
  
  cmap = matplotlib.cm.get_cmap(name='RdBu').reversed()
  # v_range = percentile(ratios_list, 90)
  # plot_pies(ratios_list, user_nums_list, -v_range, v_range, metric, output_folder + 'result.png', cmap)
  plot_pies(ratios_list, user_nums_list, -2.0, 2.0, metric, output_folder + 'result_limited.png', cmap)
  plt.close()


def plot_cut(communities: Communities, output_folder: str = ''):
  num_iters = communities.num_iters
  ratios_list = [[] for i in range(num_iters + 1)]
  user_nums_list = [[] for i in range(num_iters + 1)]

  def fun(_communities: Communities, num: int):
    # num = _communities.num_iters
    n_users = len(_communities.user_index)
    ratio = _communities.cut_ratio
    user_nums_list[num_iters - num].append(n_users)
    ratios_list[num_iters - num].append(ratio)
    # print(f'fun(num={num}, n_users={n_users}) start')
    # print('-----s0-----')
    if _communities.s0 is None:
      n0_users = len(_communities.c0.users)
      # print(f'fill s0 with n_users={n0_users}')
      for i in range(num):
        user_nums_list[num_iters - i].append(n0_users)
        ratios_list[num_iters - i].append(ratio)
    else:
      # assert len(_communities.c0.users) == len(_communities.s0.user_index)
      fun(_communities.s0, num - 1)
    # print('-----s1-----')
    if _communities.s1 is None:
      n1_users = len(_communities.c1.users)
      # print(f'fill s1 with n_users={n1_users}')
      for i in range(num):
        user_nums_list[num_iters - i].append(n1_users)
        ratios_list[num_iters - i].append(ratio)
    else:
      # assert len(_communities.c1.users) == len(_communities.s1.user_index)
      fun(_communities.s1, num - 1)
    # print(f'fun(num={num}, n_users={n_users}) end')
    # assert len(_communities.c0.users) + len(_communities.c1.users) == n_users

  fun(communities, num_iters)
  vmin = 1.0
  vmax = 0.0
  for ratios in ratios_list:
    for ratio in ratios:
      vmin = min(vmin, ratio)
      vmax = max(vmax, ratio)
  plot_pies(ratios_list, user_nums_list, vmin, vmax, 'cut', output_folder + 'cut_info.png')
  # for user_nums in user_nums_list:
    # print(user_nums)
  plt.close()


def plot_density(communities: Communities, output_folder: str = ''):
  num_iters = communities.num_iters
  ratios_list = [[] for i in range(num_iters + 1)]
  user_nums_list = [[] for i in range(num_iters + 1)]

  def fun(_communities: Communities, num: int):
    n_users = len(_communities.user_index)
    ratio = _communities.density
    user_nums_list[num_iters - num].append(n_users)
    ratios_list[num_iters - num].append(ratio)
    if _communities.s0 is None:
      n0_users = len(_communities.c0.users)
      for i in range(num):
        user_nums_list[num_iters - i].append(n0_users)
        ratios_list[num_iters - i].append(ratio)
    else:
      fun(_communities.s0, num - 1)
    if _communities.s1 is None:
      n1_users = len(_communities.c1.users)
      for i in range(num):
        user_nums_list[num_iters - i].append(n1_users)
        ratios_list[num_iters - i].append(ratio)
    else:
      fun(_communities.s1, num - 1)

  fun(communities, num_iters)
  v_range = percentile(ratios_list, 98)
  plot_pies(ratios_list, user_nums_list, 0, v_range, 'densiity', output_folder + 'density.png')
  plt.close()


def plot_divide(communities: Communities, output_folder: str = ''):
  num_iters = communities.num_iters
  ratios_list = [[] for i in range(num_iters + 1)]
  user_nums_list = [[] for i in range(num_iters + 1)]

  def fun(_communities: Communities, num: int):
    # num = _communities.num_iters
    n_users = len(_communities.user_index)
    ratio = _communities.divide_info
    user_nums_list[num_iters - num].append(n_users)
    ratios_list[num_iters - num].append(ratio)
    # print(f'fun(num={num}, n_users={n_users}) start')
    # print('-----s0-----')
    if _communities.s0 is None:
      n0_users = len(_communities.c0.users)
      # print(f'fill s0 with n_users={n0_users}')
      for i in range(num):
        user_nums_list[num_iters - i].append(n0_users)
        ratios_list[num_iters - i].append(ratio)
    else:
      # assert len(_communities.c0.users) == len(_communities.s0.user_index)
      fun(_communities.s0, num - 1)
    # print('-----s1-----')
    if _communities.s1 is None:
      n1_users = len(_communities.c1.users)
      # print(f'fill s1 with n_users={n1_users}')
      for i in range(num):
        user_nums_list[num_iters - i].append(n1_users)
        ratios_list[num_iters - i].append(ratio)
    else:
      # assert len(_communities.c1.users) == len(_communities.s1.user_index)
      fun(_communities.s1, num - 1)
    # print(f'fun(num={num}, n_users={n_users}) end')
    # assert len(_communities.c0.users) + len(_communities.c1.users) == n_users

  fun(communities, num_iters)
  v_range = percentile(ratios_list, 99)
  cmap = matplotlib.cm.get_cmap(name='RdBu').reversed()
  plot_pies(ratios_list, user_nums_list, -v_range, v_range, 'divide', output_folder + 'divide_info.png', cmap)
  plt.close()


def plot_scatter(x, Y: dict, output: str, xlabel: str = 'x', ylabel: str = 'y'):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  for key in Y:
    # plt.plot(x, Y[key], label=key)
    ax.scatter(x, Y[key], label=key, s=5)
  plt.legend() # 让图例生效
  # plt.xticks(x, names, rotation=45)
  # plt.margins(0)
  # plt.subplots_adjust(bottom=0.15)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  # plt.xlabel(xlabel) #X轴标签
  # plt.ylabel(ylabel) #Y轴标签
  # plt.title("Figure") #标题
  # tag = '_'.join(Y.keys())
  # fig.savefig(f'{output}/{ylabel}_{xlabel}_{tag}.png')
  fig.savefig(f'{output}/{ylabel}_{xlabel}.png')
  fig.clf()
  plt.close()


def plot_line(x, Y: dict, output: str, xlabel: str = 'x', ylabel: str = 'y', lim: bool = True):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  for key in Y:
    # plt.plot(x, Y[key], label=key)
    ax.plot(x, Y[key], label=key)
  plt.legend() # 让图例生效
  # plt.xticks(x, names, rotation=45)
  # plt.margins(0)
  # plt.subplots_adjust(bottom=0.15)
  # ax.set_xscale('log')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xlim(0, x[-1])
  if lim:
    if ylabel in ['MAE']:
      ax.set_ylim(0.75, 1.15)
    if ylabel in ['W-MAE']:
      ax.set_ylim(0.77, 1.17)
    elif ylabel in ['RMSE']:
      ax.set_ylim(0.95, 1.35)
    elif ylabel in ['W-RMSE']:
      ax.set_ylim(0.98, 1.38)
  # plt.title("Figure") #标题
  # plt.savefig(f'{output}/{xlabel}_{ylabel}.png')
  fig.savefig(f'{output}/{ylabel}_{xlabel}.png')
  fig.clf()
  plt.close()


def plot_line_xticks(x, Y: dict, output: str, xlabel: str = 'x', ylabel: str = 'y'):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  num = len(x)
  xi = list(range(num))
  for key in Y:
    # plt.plot(x, Y[key], label=key)
    ax.plot(xi, Y[key], label=key, marker='s', linewidth=1.0)
  plt.legend() # 让图例生效
  # plt.xticks(x, names, rotation=45)
  # plt.margins(0)
  # plt.subplots_adjust(bottom=0.15)
  # ax.set_xscale('log')
  xi = [xi[i] for i in range(0, num, 2)]
  x  = [ x[i] for i in range(0, num, 2)]
  ax.set_xticks(xi)
  ax.set_xticklabels(x)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  # plt.xlabel(xlabel) #X轴标签
  # plt.ylabel(ylabel) #Y轴标签
  # plt.title("Figure") #标题
  # plt.savefig(f'{output}/{xlabel}_{ylabel}.png')
  # tag = '_'.join(Y.keys())
  # fig.savefig(f'{output}/{ylabel}_{xlabel}_{tag}.png')
  fig.savefig(f'{output}/{ylabel}_{xlabel}.png')
  fig.clf()
  plt.close()


def plot_rating(x, y, output: str='tmp/rating.png'):
  # t-SNE的最终结果的降维与可视化
  ts_2d = manifold.TSNE(n_components=2, init='pca', random_state=0)
  x_ts = ts_2d.fit_transform(x)
  x_min, x_max = x_ts.min(0), x_ts.max(0)
  x_range = x_max - x_min
  x_range[x_range <= 0] = 1
  x_2d = (x_ts - x_min) / x_range
  ts_3d = manifold.TSNE(n_components=3, init='pca', random_state=0)
  x_ts = ts_3d.fit_transform(x)
  x_min, x_max = x_ts.min(0), x_ts.max(0)
  x_range = x_max - x_min
  x_range[x_range <= 0] = 1
  x_3d = (x_ts - x_min) / x_range
  # y \in [1, 5]
  cmap = plt.get_cmap('viridis', 6)
  color = [cmap(int(i)) for i in y]
  fig = plt.figure(figsize=(12, 6))
  plt.set_cmap(cmap)
  # print 2d
  ax_2d = fig.add_subplot(1, 2, 1)
  im_2d = ax_2d.scatter(x_2d[:,0], x_2d[:,1], s=10, c=color, marker='.')
  fig.colorbar(im_2d, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*5)))
  # print 3d
  ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
  im_3d = ax_3d.scatter(x_3d[:,0], x_3d[:,1], x_3d[:, 2], s=10, c=color, marker='.')
  fig.colorbar(im_3d, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*5)))

  fig.savefig(output)
  plt.close()
