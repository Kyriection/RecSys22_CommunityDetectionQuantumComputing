from CommunityDetection.BaseCommunityDetection import BaseCommunityDetection


class KmeansCommunityDetection(BaseCommunityDetection):
    is_qubo = False
    filter_items = False
    name = 'KmeansCommunityDetection'

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        raise NotImplementedError('KmeansCommunityDetection.__init__() not support.')


    def save_model(self, folder_path, file_name):
        raise NotImplementedError('KmeansCommunityDetection.save_model() not support.')

    def run(self):
        raise NotImplementedError('KmeansCommunityDetection.run() please use classical Kmeans.')
    
    def fit(self, *args, **kwargs):
        raise NotImplementedError('KmeansCommunityDetection.fit() please use classical Kmeans.')
