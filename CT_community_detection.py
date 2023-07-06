import shutil, os, argparse, logging

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
    QUBOBipartiteProjectedItemCommunityDetection, LTBipartiteProjectedCommunityDetection, QuantityDivision, \
    QUBOBipartiteProjectedCommunityDetection2, LTBipartiteCommunityDetection, METHOD_DICT, CascadeCommunityDetection, \
    get_cascade_class, UserBipartiteCommunityDetection, TestCommunityDetection
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, MovielensSampleReader, \
    MovielensSample2Reader, MovielensSample3Reader
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.plot import plot_cut, plot_density
from utils.urm import get_community_urm, load_data, merge_sparse_matrices, show_urm_info, head_tail_cut

logging.basicConfig(level=logging.INFO)
MIN_COMMUNITIE_SIZE = 1
CUT_RATIO: float = None
LT_METHOD = {
    QUBOBipartiteProjectedCommunityDetection: LTBipartiteProjectedCommunityDetection,
    QUBOBipartiteCommunityDetection: LTBipartiteCommunityDetection
}
A1_LAYER = 0


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
    global CUT_RATIO
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
        # dataset_folder_path = os.path.join(result_folder_path, dataset_name)
        urm_train, urm_validation, urm_test, icm, ucm = load_data(data_reader, split_quota=split_quota, user_wise=user_wise,
                                                        make_implicit=make_implicit, threshold=threshold, icm_ucm=True)
        # item is main charactor, and remove year from item comtext
        urm_train, urm_validation, urm_test, icm, ucm = urm_train.T.tocsr(), urm_validation.T.tocsr(), urm_test.T.tocsr(), ucm, icm[:, :-1]
        urm_train_last_test = merge_sparse_matrices(urm_train, urm_validation)
        urm_all = merge_sparse_matrices(urm_train_last_test, urm_test)
        _, _, _, _, _, urm_train, urm_validation, urm_test, icm, ucm = head_tail_cut(CUT_RATIO, urm_all, urm_validation, urm_test, icm, ucm)

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

    method_folder_path = f'{folder_path}{method.name}/'
    plot_density(communities, method_folder_path)


def cd_per_iter(cd_urm, icm, ucm, method, folder_path, sampler: dimod.Sampler = None, communities: Communities = None,
                n_iter: int = 0, **kwargs):
    print(f'Running community detection iteration {n_iter} with {method.name}...')
    logging.info(f'Running community detection iteration {n_iter} with {method.name}...')
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

    if n_iter < A1_LAYER:
        assert method in LT_METHOD, f'{method.name} not support quantity attribute.'
        logging.info(f'n_iter={n_iter}<{A1_LAYER}, {method.name} to {LT_METHOD[method].name}')
        lt_method = LT_METHOD[method]
        # lt_method.name = method.name
        method.name = lt_method.name
        m: BaseCommunityDetection = lt_method(cd_urm, icm, ucm)
    else:
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
    if communities is not None:
        logging.debug(f'{cd_urm.size} / {communities.n_users}')
        communities.density = cd_urm.size / communities.n_users
    return communities


def check_communities(communities: Communities, check_users, check_items):
    global MIN_COMMUNITIE_SIZE
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
    parser.add_argument('method', nargs='+', type=str, help='method',
                        choices=['QUBOBipartiteCommunityDetection', 'QUBOBipartiteProjectedCommunityDetection',
                                 'LTBipartiteCommunityDetection', 'LTBipartiteProjectedCommunityDetection',
                                 'KmeansCommunityDetection', 'QuantityDivision', 'HybridCommunityDetection',
                                 'TestCommunityDetection'])
    parser.add_argument('-c', '--cut_ratio', type=float, default=0.0, help='head ratio for clustered tail')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha for quantity')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='beta for cascade')
    parser.add_argument('-t', '--T', type=int, default=5, help='T for quantity')
    parser.add_argument('-l', '--layer', type=int, default=0, help='number of layer of quantity')
    parser.add_argument('-o', '--ouput', type=str, default='results', help='the path to save the result')
    parser.add_argument('--attribute', action='store_true', help='Use item attribute data (cascade) or not')
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
                logging.debug(f'rm {method_folder_path}')
                shutil.rmtree(method_folder_path)


if __name__ == '__main__':
    args = parse_args()
    CUT_RATIO = args.cut_ratio
    A1_LAYER = args.layer
    data_reader_classes = [Movielens100KReader]
    data_reader_classes = [MovielensSample3Reader]
    # data_reader_classes = [Movielens1MReader]
    # data_reader_classes = [Movielens100KReader, Movielens1MReader, FilmTrustReader, MovielensHetrec2011Reader,
                        #    LastFMHetrec2011Reader, FrappeReader, CiteULike_aReader, CiteULike_tReader]
    method_list = [METHOD_DICT[method_name] for method_name in args.method]
    sampler_list = [neal.SimulatedAnnealingSampler()]
    # sampler_list = [greedy.SteepestDescentSampler(), tabu.TabuSampler()]
    # sampler_list = [LeapHybridSampler()]
    # sampler_list = [LeapHybridSampler(), neal.SimulatedAnnealingSampler(), greedy.SteepestDescentSampler(),
                    # tabu.TabuSampler()]
    num_iters = 5
    result_folder_path = f'{os.path.abspath(args.ouput)}/'
    QUBOGraphCommunityDetection.set_alpha(args.alpha)
    QUBOProjectedCommunityDetection.set_alpha(args.alpha)
    HybridCommunityDetection.set_beta(args.beta)
    LTBipartiteProjectedCommunityDetection.set_alpha(args.alpha)
    LTBipartiteCommunityDetection.set_alpha(args.alpha)
    LTBipartiteProjectedCommunityDetection.set_T(args.T)
    LTBipartiteCommunityDetection.set_T(args.T)
    QuantityDivision.set_T(args.T)
    KmeansCommunityDetection.set_attribute(args.attribute)
    TestCommunityDetection.set_beta(args.beta)
    if args.attribute:
        for i, method in enumerate(method_list):
            method_list[i] = get_cascade_class(method)
            method_list[i].set_beta(args.beta)
    # clean_results(result_folder_path, data_reader_classes, method_list)
    main(data_reader_classes, method_list, sampler_list, result_folder_path, num_iters=num_iters)
