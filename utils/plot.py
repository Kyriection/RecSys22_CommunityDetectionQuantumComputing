from typing import List

import matplotlib
import matplotlib.pyplot as plt

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
