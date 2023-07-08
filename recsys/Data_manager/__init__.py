#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/04/2019

@author: Anonymous
"""

from recsys.Data_manager.Movielens.MovielensSampleReader import MovielensSampleReader
from recsys.Data_manager.Movielens.MovielensSample2Reader import MovielensSample2Reader
from recsys.Data_manager.Movielens.MovielensSample3Reader import MovielensSample3Reader
from recsys.Data_manager.Movielens.Movielens100KReader import Movielens100KReader
from recsys.Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from recsys.Data_manager.Movielens.Movielens10MReader import Movielens10MReader
from recsys.Data_manager.Movielens.Movielens20MReader import Movielens20MReader
from recsys.Data_manager.Movielens.MovielensHetrec2011Reader import MovielensHetrec2011Reader
from recsys.Data_manager.LastFMHetrec2011.LastFMHetrec2011Reader import LastFMHetrec2011Reader
from recsys.Data_manager.CiteULike.CiteULikeReader import CiteULike_aReader, CiteULike_tReader
from recsys.Data_manager.Frappe.FrappeReader import FrappeReader
from recsys.Data_manager.FilmTrust.FilmTrustReader import FilmTrustReader

DATA_LIST = [Movielens100KReader, Movielens1MReader, MovielensHetrec2011Reader,
             MovielensSampleReader, MovielensSample2Reader, MovielensSample3Reader]

DATA_DICT = {data_reader.DATASET_SUBFOLDER[:-1]: data_reader for data_reader in DATA_LIST}