'''
Author: Kaizyn
Date: 2023-01-28 22:42:44
LastEditTime: 2023-01-28 23:23:50
'''
# -*- coding: utf-8 -*-
# coding: utf-8
import os, random, zipfile

FILE_PATH = os.path.dirname(__file__)

SOURCE = "Movielens1M"
DATA_FILE_PREFIX = 'ml-1m'
NUM_USER = 6040
NUM_ITEM = 3952 # 3883
DATA_FILE = {
  'users': f"{DATA_FILE_PREFIX}/users.dat",
  'movies': f"{DATA_FILE_PREFIX}/movies.dat",
  'ratings': f"{DATA_FILE_PREFIX}/ratings.dat",
}

SOURCE = "Movielens100K"
DATA_FILE_PREFIX = 'ml-100k'
NUM_USER = 943
NUM_ITEM = 1682
DATA_FILE = {
  'users': "u.user",
  'movies': "u.item",
  'ratings': "u.data",
}

SOURCE_PATH = os.path.join(FILE_PATH, SOURCE)

output = input("Output: ")
output = output or "MovielensSample3"
output_folder = os.path.join(FILE_PATH, output)
output_path = os.path.join(output_folder, 'ml-sample')

if not os.path.exists(output_path):
  # os.mkdir(output_path)
  os.makedirs(output_path, exist_ok=True)
  # os.system(f'mkdir -r {OUTPUT_PATH}')

num_user = input("number of user: ")
num_item = input("number of item: ")
num_user = int(num_user) if num_user else 240
num_item = int(num_item) if num_item else 470
users = random.sample(range(1, NUM_USER + 1), num_user)
items = random.sample(range(1, NUM_ITEM + 1), num_item)
user_ratings = {}
item_ratings = {}
user_dict = {}
item_dict = {}
for i, id in enumerate(users):
  user_dict[str(id)] = str(i)
for i, id in enumerate(items):
  item_dict[str(id)] = str(i)

print("---unzip---")
decompressed_path = os.path.join(SOURCE_PATH, 'decompressed')
if not os.path.exists(decompressed_path):
  os.mkdir(decompressed_path)
  dataFile = zipfile.ZipFile(SOURCE_PATH + f"/{DATA_FILE_PREFIX}.zip")
  users_path = dataFile.extract(f"{DATA_FILE_PREFIX}/{DATA_FILE['users']}", path=decompressed_path)
  movies_path = dataFile.extract(f"{DATA_FILE_PREFIX}/{DATA_FILE['movies']}", path=decompressed_path)
  ratings_path = dataFile.extract(f"{DATA_FILE_PREFIX}/{DATA_FILE['ratings']}", path=decompressed_path)
else:
  users_path = os.path.join(decompressed_path, f"{DATA_FILE_PREFIX}/{DATA_FILE['users']}")
  movies_path = os.path.join(decompressed_path, f"{DATA_FILE_PREFIX}/{DATA_FILE['movies']}")
  ratings_path = os.path.join(decompressed_path, f"{DATA_FILE_PREFIX}/{DATA_FILE['ratings']}")


print("---start ratings.dat---")
cnt = 0
with open(os.path.join(output_path, DATA_FILE['ratings']), 'w', encoding='ISO-8859-1') as o:
  with open(ratings_path, 'r', encoding='ISO-8859-1') as i:
    contexts = i.readlines()
    for context in contexts:
      sep = '::' if SOURCE == 'Movielens1M' else '\t'
      context = context.split(sep, maxsplit=2)
      context[0] = user_dict.get(context[0], -1)
      context[1] = item_dict.get(context[1], -1)
      if context[0] == -1 or context[1] == -1:
        continue
      user_ratings[context[0]] = user_ratings.get(context[0], 0) + 1
      item_ratings[context[1]] = item_ratings.get(context[1], 0) + 1
    # delete user/item with no interaction
    user_dict = {}
    item_dict = {}
    num_user = 0
    num_item = 0
    for id in users:
      if user_ratings.get(str(id), 0) > 0:
        num_user += 1
        user_dict[str(id)] = str(num_user)
    for id in items:
      if item_ratings.get(str(id), 0) > 0:
        num_item += 1
        item_dict[str(id)] = str(num_item)
    # create new ratings.dat
    for context in contexts:
      context = context.split(sep, maxsplit=2)
      context[0] = user_dict.get(context[0], -1)
      context[1] = item_dict.get(context[1], -1)
      if context[0] == -1 or context[1] == -1:
        continue
      context = sep.join(context)
      o.write(context)
      cnt += 1
print("---total line: ", cnt)
print(f'num_user={num_user}, num_item={num_item}')

print("---start users.dat---")
cnt = 0
with open(os.path.join(output_path, DATA_FILE['users']), 'w', encoding='ISO-8859-1') as o:
  with open(users_path, 'r', encoding='ISO-8859-1') as i:
    for context in i.readlines():
      sep = '::' if SOURCE == 'Movielens1M' else '|'
      context = context.split(sep, maxsplit=1)
      context[0] = user_dict.get(context[0], -1)
      if context[0] == -1:
        continue
      context = sep.join(context)
      o.write(context)
      cnt += 1
print("---total line: ", cnt)

print("---start movie.dat---")
cnt = 0
with open(os.path.join(output_path, DATA_FILE['movies']), 'w', encoding='ISO-8859-1') as o:
  with open(movies_path, 'r', encoding='ISO-8859-1') as i:
    for context in i.readlines():
      sep = '::' if SOURCE == 'Movielens1M' else '|'
      context = context.split(sep, maxsplit=1)
      context[0] = item_dict.get(context[0], -1)
      if context[0] == -1:
        continue
      context = sep.join(context)
      o.write(context)
      cnt += 1
print("---total line: ", cnt)

print('---zip start---')
with zipfile.ZipFile(os.path.join(output_folder, 'ml-sample.zip'), 'w') as z:
  z.write(output_path, arcname=(dn := os.path.basename(output_path)))
  for file_name in os.listdir(output_path):
    file_path = os.path.join(output_path, file_name)
    z.write(
      fp := file_path,
      arcname=dn + '/' + os.path.relpath(fp, output_path)
    )