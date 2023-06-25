from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

from recsys.Recommenders.SklearnRecommender import get_recommender_class

LRRecommender = get_recommender_class(LinearRegression, 'LRRecommender')
SVRRecommender = get_recommender_class(SVR, 'SVRRecommender')
DTRecommender = get_recommender_class(DecisionTreeRegressor, 'DTRecommender')