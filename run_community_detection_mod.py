import shutil, os, argparse

import dimod
import greedy
import neal
import numpy as np
import tabu
import pandas as pd
from dwave.system import LeapHybridSampler

from CommunityDetection import BaseCommunityDetection, QUBOCommunityDetection, QUBOBipartiteCommunityDetection, \
    QUBOBipartiteProjectedCommunityDetection, Communities, Community, get_community_folder_path, EmptyCommunityError, \
    UserCommunityDetection, KmeansCommunityDetection, HierarchicalClustering, QUBOGraphCommunityDetection, \
    QUBOProjectedCommunityDetection, HybridCommunityDetection, QUBONcutCommunityDetection, SpectralClustering, \
    QUBOBipartiteProjectedItemCommunityDetection
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, MovielensSampleReader, \
    MovielensSample2Reader
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.plot import plot_cut, plot_density
from utils.urm import get_community_urm, load_data, merge_sparse_matrices, show_urm_info

CUT_FLAG = False
cut_info = ['n_iter', 'n_users', 'n_items', 'c0_users', 'c0_items', 'c1_users', 'c1_items', 'cut_weight', 'all_weight', 'cut_ratio']
cut_data = {key: [] for key in cut_info}

def save_cut(cut: float, all: float, n_iter: int, communities: Communities):
    cut_data['n_iter'].append(n_iter)
    cut_data['n_users'].append(communities.n_users)
    cut_data['c0_users'].append(len(communities.c0.users))
    cut_data['c1_users'].append(len(communities.c1.users))
    cut_data['n_items'].append(communities.n_items)
    cut_data['c0_items'].append(len(communities.c0.items))
    cut_data['c1_items'].append(len(communities.c1.items))
    cut_data['cut_weight'].append(cut)
    cut_data['all_weight'].append(all)
    cut_data['cut_ratio'].append(cut / all)
    communities.cut_ratio = cut / all
    communities.density = all / communities.n_users


def load_communities(folder_path, method, sampler=None, n_iter=0, n_comm=None):
    method_folder_path = f'{folder_path}{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}/'

    try:
        communities = Communities.load(method_folder_path, 'communities', n_iter=n_iter, n_comm=n_comm,
                                       folder_suffix=folder_suffix)
        print(f'Loaded previously computed communities for {communities.num_iters + 1} iterations.')
    except FileNotFoundError:
        print('No communities found to load. Computing new communities...')
        communities = None
    return communities


def main(data_reader_classes, method_list: Iterable[Type[BaseCommunityDetection]],
         sampler_list: Iterable[dimod.Sampler], result_folder_path: str, num_iters: int = 3):
    split_quota = [80, 10, 10]
    user_wise = False
    make_implicit = True
    threshold = None

    fit_args = {
        'threshold': None,
    }

    sampler_args = {
        'num_reads': 100,
    }

    save_model = True

    for data_reader_class in data_reader_classes:
        data_reader = data_reader_class()
        dataset_name = data_reader._get_dataset_name()
        dataset_folder_path = f'{result_folder_path}{dataset_name}/'
        urm_train, urm_validation, urm_test, icm, ucm = load_data(data_reader, split_quota=split_quota, user_wise=user_wise,
                                                        make_implicit=make_implicit, threshold=threshold, icm_ucm=True)

        urm_train_last_test = merge_sparse_matrices(urm_train, urm_validation)

        for method in method_list:
            cd_per_method(urm_train_last_test, icm, ucm, method, sampler_list, dataset_folder_path, num_iters=num_iters,
                          fit_args=fit_args, sampler_args=sampler_args, save_model=save_model)


def cd_per_method(cd_urm, icm, ucm, method, sampler_list, folder_path, num_iters=1, **kwargs):
    if method.is_qubo:
        for sampler in sampler_list:
            community_detection(cd_urm, icm, ucm, method, folder_path, sampler=sampler, num_iters=num_iters,
                                **kwargs)
    else:
        community_detection(cd_urm, icm, ucm, method, folder_path, num_iters=num_iters, **kwargs)


def community_detection(cd_urm, icm, ucm, method, folder_path, sampler: dimod.Sampler = None, num_iters: int = 1, **kwargs):
    communities = load_communities(folder_path, method, sampler)
    starting_iter = 0 if communities is None else communities.num_iters + 1
    for n_iter in range(starting_iter, num_iters):
        try:
            communities = cd_per_iter(cd_urm, icm, ucm, method, folder_path, sampler=sampler, communities=communities,
                                      n_iter=n_iter, **kwargs)
        except EmptyCommunityError as e:
            print(e)
            print(f'Stopping at iteration {n_iter}.')
            clean_empty_iteration(n_iter, folder_path, method, sampler=sampler)
            break
    print("---------community_detection end ---------")
    if communities is None:
        return
    print(f"communities.num_iters={communities.num_iters}")
    if not CUT_FLAG:
        return
    df = pd.DataFrame(cut_data)
    output_path = os.path.join(folder_path, method.name, 'cut.csv')
    df.to_csv(output_path)
    method_folder_path = f'{folder_path}{method.name}/'
    plot_cut(communities, method_folder_path)
    plot_density(communities, method_folder_path)


def cd_per_iter(cd_urm, icm, ucm, method, folder_path, sampler: dimod.Sampler = None, communities: Communities = None,
                n_iter: int = 0, **kwargs):
    print(f'Running community detection iteration {n_iter} with {method.name}...')
    if communities is None:
        assert n_iter == 0, 'If no communities are given this must be the first iteration.'

        communities = run_cd(cd_urm, icm, ucm, method, folder_path, sampler=sampler, n_iter=n_iter, n_comm=None, **kwargs)
        if communities is None:
            raise EmptyCommunityError('Empty community found.')
    else:
        assert n_iter != 0, 'Cannot be the first iteration if previously computed communities are given.'

        empty_communities_flag = True
        new_communities = []
        n_comm = 0
        for community in communities.iter(n_iter):
            cd = run_cd(cd_urm, icm, ucm, method, folder_path, sampler=sampler, community=community, n_iter=n_iter, n_comm=n_comm,
                        **kwargs)
            if cd is not None:
                empty_communities_flag = False
            new_communities.append(cd)
            n_comm += 1
        if empty_communities_flag:
            raise EmptyCommunityError('Empty communities found.')
        used = communities.add_iteration(new_communities)
        assert used == len(new_communities), "commuities.add_iteration error, used items not equal to input."

    print('Saving community detection results...')
    method_folder_path = f'{folder_path}{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}/'
    communities.save(method_folder_path, 'communities', folder_suffix=folder_suffix)

    return communities


def run_cd(cd_urm, icm, ucm, method: Type[BaseCommunityDetection], folder_path: str, sampler: dimod.Sampler = None,
           community: Community = None, n_iter: int = 0, n_comm: int = None, **kwargs) -> Communities:
    n_users, n_items = cd_urm.shape
    user_index = np.arange(n_users)
    item_index = np.arange(n_items)

    if community is not None:
        cd_urm, user_index, item_index, icm, ucm = get_community_urm(cd_urm, community, filter_users=method.filter_users,
                                                           filter_items=method.filter_items, remove=True, icm=icm, ucm=ucm)
    n_users, n_items = cd_urm.shape
    show_urm_info(cd_urm)

    m: BaseCommunityDetection = method(cd_urm, icm, ucm)

    method_folder_path = f'{folder_path}{m.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}/'
    method_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, folder_suffix=folder_suffix)

    comm_file_suffix = f'{n_comm:02d}' if n_comm is not None else ''
    model_file_name = f'model{comm_file_suffix}'

    try:
        m.load_model(method_folder_path, model_file_name)
        print('Loaded previously computed CD model.')
    except FileNotFoundError:
        fit_args = kwargs.get('fit_args', {})
        m.fit(**fit_args)

        if kwargs.get('save_model', True):
            print('Saving CD model...')
            m.save_model(method_folder_path, model_file_name)

    dataIO = DataIO(method_folder_path)
    run_file_name = f'run{comm_file_suffix}'

    try:
        run_dict = dataIO.load_data(run_file_name)
        if sampler is not None:
            assert method.is_qubo, 'Cannot use a QUBO sampler on a non-QUBO method.'
            m: QUBOCommunityDetection
            sampleset = dimod.SampleSet.from_serializable(run_dict['sampleset'])
            users, items = m.get_comm_from_sample(sampleset.first.sample, n_users, n_items=n_items)
        else:
            users = run_dict['users']
            items = run_dict['items']
        print(f'Loaded previous CD run {n_comm:02d}.')

    except FileNotFoundError:
        print('Running CD...')
        if sampler is not None:
            assert method.is_qubo, 'Cannot use a QUBO sampler on a non-QUBO method.'
            m: QUBOCommunityDetection

            sampler_args = kwargs.get('sampler_args', {})
            sampleset, sampler_info, run_time = m.run(sampler, sampler_args)

            data_dict_to_save = {
                'sampleset': sampleset.to_serializable(),
                'sampler_info': sampler_info,
                'run_time': run_time,
            }
            users, items = m.get_comm_from_sample(sampleset.first.sample, n_users, n_items=n_items)
        else:
            users, items, run_time = m.run()

            data_dict_to_save = {
                'users': users,
                'items': items,
                'run_time': run_time,
            }

        dataIO.save_data(run_file_name, data_dict_to_save)
    
    communities = Communities(users, items, user_index, item_index)
    # check_communities(communities, m.filter_users, m.filter_items)
    # return communities
    # return check_communities(communities, m.filter_users, m.filter_items)
    communities = check_communities(communities, m.filter_users, m.filter_items)
    if CUT_FLAG and communities is not None:
        cut, all = m.get_graph_cut(communities)
        save_cut(cut, all, n_iter, communities)
    return communities


def check_communities(communities: Communities, check_users, check_items):
    MIN_COMMUNITIE_SIZE = 5
    for community in communities.iter():
        if (check_users and community.users.size == 0) or (check_items and community.items.size == 0):
            # raise EmptyCommunityError('Empty community found.')
            print('Empty community found.')
            return None
        if (check_users and community.users.size < MIN_COMMUNITIE_SIZE) or (check_items and community.items.size < MIN_COMMUNITIE_SIZE):
            print(f'Community size too small: user: {community.users.size}, item: {community.items.size}.')
            return None
    return communities


def clean_empty_iteration(n_iter: int, folder_path: str, method: Type[BaseCommunityDetection],
                          sampler: dimod.Sampler = None):
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}/'
    folder_path = f'{folder_path}{method.name}/'
    rm_folder_path = get_community_folder_path(folder_path, n_iter=n_iter, folder_suffix=folder_suffix)
    shutil.rmtree(rm_folder_path)

    try:
        communities = Communities.load(folder_path, 'communities', n_iter=0, folder_suffix=folder_suffix)
        print(f'Reloaded previously computed communities for {communities.num_iters + 1} iterations.')
        communities.save(folder_path, 'communities', n_iter=0, folder_suffix=folder_suffix)
        print('Saved the cleaned communities.')
    except FileNotFoundError:
        print('Cannot load communities, cleaning not complete.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0.5)
    args = parser.parse_args()
    return args


def clean_results(result_folder_path, data_reader_classes, method_list):
    for data_reader_class in data_reader_classes:
        data_reader = data_reader_class()
        dataset_name = data_reader._get_dataset_name()
        dataset_folder_path = f'{result_folder_path}{dataset_name}/'
        for method in method_list:
            method_folder_path = f'{dataset_folder_path}{method.name}/'
            if os.path.exists(method_folder_path):
                shutil.rmtree(method_folder_path)

if __name__ == '__main__':
    args = parse_args()
    data_reader_classes = [MovielensSampleReader]
    # data_reader_classes = [Movielens1MReader]
    # data_reader_classes = [Movielens100KReader, Movielens1MReader, FilmTrustReader, MovielensHetrec2011Reader,
                        #    LastFMHetrec2011Reader, FrappeReader, CiteULike_aReader, CiteULike_tReader]
    # method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection, UserCommunityDetection]
    method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection]
    # method_list = [QUBOBipartiteCommunityDetection]
    sampler_list = [neal.SimulatedAnnealingSampler()]
    # sampler_list = [greedy.SteepestDescentSampler(), tabu.TabuSampler()]
    # sampler_list = [LeapHybridSampler()]
    # sampler_list = [LeapHybridSampler(), neal.SimulatedAnnealingSampler(), greedy.SteepestDescentSampler(),
                    # tabu.TabuSampler()]
    num_iters = 7
    result_folder_path = './results/'
    clean_results(result_folder_path, data_reader_classes, method_list)
    # QUBOGraphCommunityDetection.set_alpha(args.alpha)
    # QUBOProjectedCommunityDetection.set_alpha(args.alpha)
    # HybridCommunityDetection.set_alpha(1/64)
    main(data_reader_classes, method_list, sampler_list, result_folder_path, num_iters=num_iters)
