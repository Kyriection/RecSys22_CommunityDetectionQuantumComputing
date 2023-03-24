import matplotlib
import matplotlib.pyplot as plt

from CommunityDetection.Communities import Communities

def plot_metric(communities: Communities, output_folder: str = '',
         cutoff: int = 10, metric: str = 'NDCG'):
  num_iters = communities.num_iters
  print(f'num_iters={num_iters}')
  ratios_list = []
  user_nums_list = []
  vmax = 1.0
  for i in range(num_iters - 1, -1, -1):
    ratios = []
    user_nums = []
    for community in communities.iter(i):
      value_test = community.result_dict_test['result_df'].loc[cutoff, metric]
      value_baseline = community.result_dict_baseline['result_df'].loc[cutoff, metric]
      if value_baseline > 0:
        ratio = (value_test - value_baseline) / value_baseline
      else:
        # ratio = -2.0 # INF tag
        ratio = 0.0
      ratios.append(ratio)
      user_nums.append(len(community.users))
      vmax = max(vmax, ratio)
      # print(f'value_test={value_test}, value_baseline={value_baseline}, ratio={ratio}.')
    ratios_list.append(ratios)
    user_nums_list.append(user_nums)

  def fun(v_range, file_name: str):
    fig, ax = plt.subplots()
    cmap = matplotlib.cm.get_cmap(name='RdBu').reversed()
    norm = matplotlib.colors.Normalize(vmin=-v_range, vmax=v_range)
    smap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(smap, label=f'{metric} % ratio')
    index = 0
    for i in range(num_iters - 1, -1, -1):
      ratios = ratios_list[index]
      user_nums = user_nums_list[index]
      index += 1
      for j, ratio in enumerate(ratios):
        if ratio < -1.5: # INF tag
          ratios[j] = vmax

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

  # plt.show()
  fun(vmax, output_folder + 'result.png')
  fun(2.0, output_folder + 'result_limited.png')