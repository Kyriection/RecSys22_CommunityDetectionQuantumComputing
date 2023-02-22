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
NUM_USER = 3952
NUM_ITEM = 6040
SOURCE_PATH = os.path.join(FILE_PATH, SOURCE)
OUTPUT_PATH = os.path.join(FILE_PATH, "MovielensSample")

# num_user = int(input("number of user: "))
# num_item = int(input("number of item: "))
num_user = 1000
num_item = 2500

users = random.sample(range(1, NUM_USER + 1), num_user)
items = random.sample(range(1, NUM_ITEM + 1), num_item)

user_dict = {}
item_dict = {}

for i, id in enumerate(users):
  user_dict[str(id)] = str(i)
for i, id in enumerate(items):
  item_dict[str(id)] = str(i)

print("---unzip---")
dataFile = zipfile.ZipFile(SOURCE_PATH + "/ml-1m.zip")
users_path = dataFile.extract("ml-1m/users.dat", path= SOURCE_PATH + "/decompressed/")
movies_path = dataFile.extract("ml-1m/movies.dat", path= SOURCE_PATH + "/decompressed/")
ratings_path = dataFile.extract("ml-1m/ratings.dat", path= SOURCE_PATH + "/decompressed/")

print("---start users.dat---")
cnt = 0
with open(os.path.join(OUTPUT_PATH, "users.dat"), 'w', encoding='ISO-8859-1') as o:
  with open(users_path, 'r', encoding='ISO-8859-1') as i:
    for context in i.readlines():
      context = context.split('::', maxsplit=1)
      context[0] = user_dict.get(context[0], -1)
      if context[0] == -1:
        continue
      context = '::'.join(context)
      o.write(context)
      cnt += 1
print("---total line: ", cnt)

print("---start movie.dat---")
cnt = 0
with open(os.path.join(OUTPUT_PATH, "movies.dat"), 'w', encoding='ISO-8859-1') as o:
  with open(movies_path, 'r', encoding='ISO-8859-1') as i:
    for context in i.readlines():
      context = context.split('::', maxsplit=1)
      context[0] = item_dict.get(context[0], -1)
      if context[0] == -1:
        continue
      context = '::'.join(context)
      o.write(context)
      cnt += 1
print("---total line: ", cnt)

print("---start ratings.dat---")
cnt = 0
with open(os.path.join(OUTPUT_PATH, "ratings.dat"), 'w', encoding='ISO-8859-1') as o:
  with open(ratings_path, 'r', encoding='ISO-8859-1') as i:
    for context in i.readlines():
      context = context.split('::', maxsplit=2)
      context[0] = user_dict.get(context[0], -1)
      context[1] = item_dict.get(context[1], -1)
      if context[0] == -1 or context[1] == -1:
        continue
      context = '::'.join(context)
      o.write(context)
      cnt += 1
print("---total line: ", cnt)