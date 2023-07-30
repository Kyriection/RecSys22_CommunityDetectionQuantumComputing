import shutil, time, os, argparse, logging, tqdm

import dimod
import minorminer
import numpy as np
from dwave.system import DWaveSampler, LeapHybridSampler, FixedEmbeddingComposite
from greedy import SteepestDescentSampler
from neal import SimulatedAnnealingSampler
from tabu import TabuSampler

from CommunityDetection import BaseCommunityDetection, QUBOCommunityDetection, QUBOBipartiteCommunityDetection, \
    QUBOBipartiteProjectedCommunityDetection, Communities, Community, get_community_folder_path, EmptyCommunityError, \
    UserCommunityDetection, HierarchicalClustering, QUBOGraphCommunityDetection, KmeansBipartiteCommunityDetection, \
    QUBOProjectedCommunityDetection, HybridCommunityDetection, QUBONcutCommunityDetection, SpectralClustering, \
    QUBOBipartiteProjectedItemCommunityDetection, LTBipartiteProjectedCommunityDetection, QuantityDivision, \
    QUBOBipartiteProjectedCommunityDetection2, LTBipartiteCommunityDetection, METHOD_DICT, CascadeCommunityDetection, \
    get_cascade_class, UserBipartiteCommunityDetection, TestCommunityDetection
from kfold_LT_community_detection import check_communities
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader, MovielensSampleReader, \
    MovielensSample2Reader, MovielensSample3Reader, DATA_DICT
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.plot import plot_cut, plot_density
from utils.urm import get_community_urm, load_data_k_fold, merge_sparse_matrices, show_urm_info, head_tail_cut_k_fold
import utils.seed


logging.basicConfig(level=logging.INFO)
MIN_COMMUNITIE_SIZE = 1


def load_communities(folder_path, method, sampler: Type[dimod.Sampler] = None, n_iter=0, n_comm=None):
    method_folder_path = f'{folder_path}{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__name__}/'

    try:
        communities = Communities.load(method_folder_path, 'communities', n_iter=n_iter, n_comm=n_comm,
                                       folder_suffix=folder_suffix)
        print(f'Loaded previously computed communities for {communities.num_iters + 1} iterations.')
    except FileNotFoundError:
        print('No communities found to load.')
        communities = None
    return communities


def main(data_reader_classes, method_list: Iterable[Type[BaseCommunityDetection]],
         base_sampler_list: Iterable[Type[dimod.Sampler]], sampler_list: Iterable[dimod.Sampler],
         results_folder_path: str, num_iters: int, n_folds: int, implicit: bool):
    # user_wise = True
    user_wise = False
    make_implicit = implicit
    threshold = None
    tag = 'cluster'

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
        for k in range(n_folds):
            logging.info(f"---------start {k}'th fold------------")
            result_folder_path = f'{results_folder_path}fold-{k:02d}/'
            dataset_folder_path = f'{result_folder_path}{dataset_name}/'

            urm_train, urm_test, icm, ucm = load_data_k_fold(tag, data_reader, user_wise=user_wise,make_implicit=make_implicit,
                                                             threshold=threshold, icm_ucm=True, n_folds=n_folds, k=k)

            # item is main charactor, and remove year from item comtext
            urm_train, urm_test, icm, ucm = urm_train.T.tocsr(), urm_test.T.tocsr(), ucm, icm

            for method in method_list:
                if 'Cascade' in method.name: method.n_all_users = urm_train.shape[0]
                qa_cd_per_method(urm_train, icm, ucm, method, base_sampler_list, sampler_list, dataset_folder_path,
                                 num_iters=num_iters, fit_args=fit_args, sampler_args=sampler_args, save_model=save_model)


def qa_cd_per_method(cd_urm, icm, ucm, method, base_sampler_list, sampler_list, folder_path, num_iters=1, **kwargs):
    if method.is_qubo:
        for base_sampler in base_sampler_list:
            for sampler in sampler_list:
                qa_community_detection(cd_urm, icm, ucm, method, folder_path, base_sampler=base_sampler, sampler=sampler,
                                       num_iters=num_iters, **kwargs)
    else:
        for sampler in sampler_list:
            qa_community_detection(cd_urm, icm, ucm, method, folder_path, sampler=sampler, num_iters=num_iters, **kwargs)


def qa_community_detection(cd_urm, icm, ucm, method, folder_path, base_sampler: Type[dimod.Sampler] = None,
                           sampler: dimod.Sampler = None, num_iters: int = 1, **kwargs):
    communities: Communities = load_communities(folder_path, method, base_sampler)
    if communities is None:
        print(f'Could not load a community for {folder_path}, {method.name}, {base_sampler.__name__}')
        return

    starting_iter = None
    for n_iter in range(num_iters):
        if check_communities_size(communities, n_iter, method.filter_users, method.filter_items):
            starting_iter = n_iter
            if 'QUBOBipartiteProjectedCommunityDetection' in method.name:
                starting_iter += 1
            break
    logging.info(f'starting_iter={starting_iter}')
    # starting_iter = 5 #################################################
    if starting_iter is None:
        print(f'Could not find a suitable iteration in range of {num_iters} iterations.')
        return

    original_iters = communities.num_iters
    if starting_iter == 0:
        communities = None
    else:
        communities.reset_from_iter(starting_iter)

    for n_iter in range(starting_iter, num_iters):
        try:
            communities = qa_cd_per_iter(cd_urm, icm, ucm, method, folder_path, base_sampler=base_sampler,
                                         sampler=sampler, communities=communities, n_iter=n_iter, **kwargs)
        except EmptyCommunityError as e:
            print(e)
            print(f'Stopping at iteration {n_iter}.')
            clean_empty_iteration(n_iter, folder_path, method, base_sampler=base_sampler, sampler=sampler)
            break
        except ValueError as e:
            print(e)
            print(f'Could not find an embedding at iteration {n_iter}.')
            clean_empty_iteration(n_iter, folder_path, method, base_sampler=base_sampler, sampler=sampler)
            if n_iter > original_iters:
                break
            continue
    print("---------community_detection end ---------")
    if communities is None:
        return
    print(f"communities.num_iters={communities.num_iters}")

    method_folder_path = f'{folder_path}{method.name}/'
    plot_density(communities, method_folder_path)


def qa_cd_per_iter(cd_urm, icm, ucm, method, folder_path, base_sampler: Type[dimod.Sampler] = None,
                   sampler: dimod.Sampler = None, communities: Communities = None, n_iter: int = 0, **kwargs):
    print(f'Running community detection iteration {n_iter} with {method.name}...')
    logging.info(f'Running community detection iteration {n_iter} with {method.name}...')
    if communities is None:
        assert n_iter == 0, 'If no communities are given this must be the first iteration.'

        communities = qa_run_cd(cd_urm, icm, ucm, method, folder_path, base_sampler=base_sampler, sampler=sampler,
                                n_iter=n_iter, n_comm=None, **kwargs)
    else:
        assert n_iter != 0, 'Cannot be the first iteration if previously computed communities are given.'

        empty_communities_flag = True
        new_communities = []
        n_comm = 0
        for community in tqdm.tqdm(communities.iter(n_iter), f'communities.iter({n_iter})'):
            cd = qa_run_cd(cd_urm, icm, ucm, method, folder_path, base_sampler=base_sampler, sampler=sampler,
                           community=community, n_iter=n_iter, n_comm=n_comm, **kwargs)
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
    folder_suffix = get_folder_suffix(base_sampler, sampler)

    try:
        communities.save_from_iter(n_iter, method_folder_path, 'communities', folder_suffix=folder_suffix)
    except Exception as e:
        logging.warning(e)
        communities.save(method_folder_path, 'communities', folder_suffix=folder_suffix)

    return communities


def qa_run_cd(cd_urm, icm, ucm, method: Type[BaseCommunityDetection], folder_path: str,
              base_sampler: Type[dimod.Sampler] = None, sampler: dimod.Sampler = None, community: Community = None,
              n_iter: int = 0, n_comm: int = None, **kwargs) -> Communities:
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
    folder_suffix = get_folder_suffix(base_sampler, sampler)
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
        # print(f'Loaded previous CD run {n_comm:02d}.')
        print(f'Loaded previous CD run {n_comm}.')

    except FileNotFoundError:
        print('Running CD...')
        if sampler is not None:
            assert method.is_qubo, 'Cannot use a QUBO sampler on a non-QUBO method.'
            m: QUBOCommunityDetection

            embedding = {}
            embedding_time = 0
            if isinstance(sampler, DWaveSampler):
                # from recsys.Data_manager.DataReader_utils import compute_density
                # logging.info(f'start finding embedding, shape={cd_urm.shape}, size={cd_urm.size}, density={compute_density(cd_urm)}.')
                embedding_time = time.time()
                # embedding = minorminer.find_embedding(m.get_Q_adjacency(), sampler.edgelist, timeout=60)
                embedding = minorminer.find_embedding(m.get_Q_adjacency(), sampler.edgelist)
                embedding_time = time.time() - embedding_time
                # logging.info(f'n_comm={n_comm}, embedding_time={embedding_time}, embedding={bool(embedding)}')
                # return

                sampler = FixedEmbeddingComposite(sampler, embedding)

            sampler_args = kwargs.get('sampler_args', {})
            sampleset, sampler_info, run_time = m.run(sampler, sampler_args)

            data_dict_to_save = {
                'sampleset': sampleset.to_serializable(),
                'sampler_info': sampler_info,
                'run_time': run_time,
            }

            if isinstance(sampler, FixedEmbeddingComposite):
                data_dict_to_save['embedding'] = embedding
                data_dict_to_save['embedding_time'] = embedding_time

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
    communities = check_communities(communities, m.filter_users, m.filter_items)
    if communities is not None:
        logging.debug(f'{cd_urm.size} / {communities.n_users}')
        communities.density = cd_urm.size / communities.n_users
    return communities


def get_folder_suffix(base_sampler, sampler):
    folder_suffix = f'{sampler.__class__.__name__}/'
    if base_sampler is not None:
        folder_suffix = f'{base_sampler.__name__}_{folder_suffix}'
    return folder_suffix


def check_communities_size(communities: Communities, n_iter: int, check_users, check_items):
    if communities is None:
        print('Non-existing communities. Please use non-null ones.')
        return False

    if communities.num_iters < n_iter:
        return False

    for community in communities.iter(n_iter):
        n_nodes = len(community.users) if check_users else 0
        n_nodes += len(community.items) if check_items else 0
        if n_nodes > 170:
            print(f'Communities size exceeded maximum size for the Quantum Annealer at iteration {n_iter}.')
            return False
    return True


def clean_empty_iteration(n_iter: int, folder_path: str, method: Type[BaseCommunityDetection],
                          base_sampler: Type[dimod.Sampler], sampler: dimod.Sampler = None, start_iter: int = 0):
    folder_suffix = get_folder_suffix(base_sampler, sampler)
    folder_path = f'{folder_path}{method.name}/'
    rm_folder_path = get_community_folder_path(folder_path, n_iter=n_iter, folder_suffix=folder_suffix)
    shutil.rmtree(rm_folder_path)

    base_folder_suffix = '' if base_sampler is None else f'{base_sampler.__class__.__name__}/'
    try:
        communities = Communities.load(folder_path, 'communities', n_iter=0, folder_suffix=base_folder_suffix)

        communities.load_from_iter(start_iter, folder_path, 'communities', folder_suffix=folder_suffix)
        print(f'Reloaded previously computed communities for {communities.num_iters + 1} iterations.')

        communities.save_from_iter(start_iter, folder_path, 'communities', folder_suffix=folder_suffix)
        print('Saved the cleaned communities.')
    except FileNotFoundError:
        print('Cannot load communities, cleaning not complete.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', nargs='+', type=str, help='method',
                        choices=['QUBOBipartiteCommunityDetection', 'QUBOBipartiteProjectedCommunityDetection',
                                 'LTBipartiteCommunityDetection', 'LTBipartiteProjectedCommunityDetection',
                                 'KmeansBipartiteCommunityDetection', 'HybridCommunityDetection',
                                 'QuantityDivision', 'QUBOBipartiteProjectedCommunityDetection2',
                                 'TestCommunityDetection'])
    parser.add_argument('-d', '--dataset', nargs='+', type=str, default=['Movielens100K'], help='dataset',
                        choices=['Movielens100K', 'Movielens1M', 'MovielensHetrec2011', 'MovielensSample',
                                 'MovielensSample2', 'MovielensSample3'])
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha for quantity')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='beta for cascade')
    parser.add_argument('-t', '--T', type=int, default=5, help='T for quantity')
    parser.add_argument('-o', '--ouput', type=str, default='results', help='the path to save the result')
    parser.add_argument('-k', '--kfolds', type=int, default=5, help='number of folds for dataset split.')
    parser.add_argument('--attribute', action='store_true', help='Use item attribute data (cascade) or not')
    parser.add_argument('--implicit', action='store_true', help='URM make implicit (values to 0/1) or not.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # data_reader_classes = [Movielens100KReader, Movielens1MReader, FilmTrustReader, MovielensHetrec2011Reader,
                        #    LastFMHetrec2011Reader, FrappeReader, CiteULike_aReader, CiteULike_tReader]
    data_reader_classes = [DATA_DICT[data_name] for data_name in args.dataset]
    # method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection]
    method_list = [METHOD_DICT[method_name] for method_name in args.method]
    # base_sampler_list = [LeapHybridSampler, SimulatedAnnealingSampler, SteepestDescentSampler, TabuSampler]
    base_sampler_list = [SimulatedAnnealingSampler]
    sampler_list = [DWaveSampler(solver={'topology__type': 'pegasus'})]
    num_iters = 10
    results_folder_path = f'{os.path.abspath(args.ouput)}/'
    QUBOGraphCommunityDetection.set_alpha(args.alpha)
    QUBOProjectedCommunityDetection.set_alpha(args.alpha)
    HybridCommunityDetection.set_beta(args.beta)
    LTBipartiteProjectedCommunityDetection.set_alpha(args.alpha)
    LTBipartiteCommunityDetection.set_alpha(args.alpha)
    LTBipartiteProjectedCommunityDetection.set_T(args.T)
    LTBipartiteCommunityDetection.set_T(args.T)
    QuantityDivision.set_T(args.T)
    KmeansBipartiteCommunityDetection.set_attribute(args.attribute)
    if args.attribute:
        for i, method in enumerate(method_list):
            method_list[i] = get_cascade_class(method)
            method_list[i].set_beta(args.beta)
    main(data_reader_classes, method_list, base_sampler_list, sampler_list, results_folder_path, num_iters=num_iters,
         n_folds=args.kfolds, implicit=args.implicit)
