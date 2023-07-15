from CommunityDetection.BaseCommunityDetection import BaseCommunityDetection


class EachItem(BaseCommunityDetection):
    is_qubo = False
    filter_items = False
    name = 'EachItem'

    def __init__(self, urm, icm, ucm, *args, **kwargs):
        raise NotImplementedError('EachItem.__init__() not support.')


    def save_model(self, folder_path, file_name):
        raise NotImplementedError('EachItem.save_model() not support.')

    def run(self):
        raise NotImplementedError('EachItem.run() please use classical Kmeans.')
    
    def fit(self, *args, **kwargs):
        raise NotImplementedError('EachItem.fit() please use classical Kmeans.')
