'''
Author: Kaizyn
Date: 2023-01-21 10:19:24
LastEditTime: 2023-01-21 10:55:37
'''
import os, zipfile
import pandas as pd

QUBO = ["Hybrid", "QUBOBipartiteCommunityDetection", "QUBOBipartiteProjectedCommunityDetection", "KmeansCommunityDetection", "HierarchicalClustering", "UserCommunityDetection"]
METHOD = ["LeapHybridSampler", "SimulatedAnnealingSampler", "SteepestDescentSolver", "TabuSampler", ""]
CDR = "cd_TopPopRecommender.zip"
RESULT = ".result_df.csv"
COL = ["PRECISION", "MAP", "NDCG", "COVERAGE_ITEM_HIT"]
CUTOFF = 10
PD_COL = ["Baseline", "N", "C", *COL]

# print(__file__)

def print_result(dataset: str = None, show: bool = True, output_folder: str = None):
  dataset = dataset or "MovielensSample"
  # dataset = "./" + dataset
  dataset = os.path.abspath("/app/results/" + dataset)

  # for method in QUBO:
  for method in os.listdir(dataset):
    path = os.path.join(dataset, method)
    if not os.path.exists(path) or os.path.isfile(path) or method == "TopPopRecommender":
      continue
    if show:
      print(method)
      # print("N", COL)
    # print(path)
    data = {key: [] for key in PD_COL}
    dir_file = os.listdir(path)
    dir_file.sort()
    for m in METHOD:
      # if show:
        # print(m)
      try:
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
          C = 0
          for c in os.listdir(tmp):
            if os.path.isdir(os.path.join(tmp, c)) and c[0] == 'c':
              try:
                C = max(C, int(c[1:]))
              except:
                continue
          file = os.path.join(tmp, CDR)
          # print(file)
          z = zipfile.ZipFile(file, 'r') 
          file_path = z.extract(RESULT, path=cur + "/decompressed")
          result = pd.read_csv(file_path, index_col="cutoff")
          result = [round(result[col][CUTOFF], 4) for col in COL]
          # if show:
            # print(name, result)
          data['Baseline'].append(m)
          data['N'].append(int(name[4:]) + 1)
          data['C'].append(C + 1)
          for i in range(len(COL)):
            data[COL[i]].append(result[i])

      except Exception as e:
        print(e)
    
    df = pd.DataFrame(data)
    if output_folder is None:
      output_folder = path
    output_path = os.path.join(output_folder, f'{method}.csv')
    df.to_csv(output_path)
    if show:
      print(df)

if __name__ == '__main__':
  dataset = input("input file folder name: ")
  show = input("print on CMD or not: ")
  show = True if show else False
  print_result(dataset, show)
  