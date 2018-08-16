from gensim.models.keyedvectors import Word2VecKeyedVectors
from scipy.spatial.distance import euclidean, cosine

from distance_metric import DistanceMetric


class PairsEvaluator(object):
    __DISTANCE_THRESHOLD = 0.35
    __DISTANCE_FUNCTIONS = {DistanceMetric.EUCLIDEAN: euclidean, DistanceMetric.COSINE: cosine}

    def __init__(self, model: Word2VecKeyedVectors, source_word: str, target_word: str) -> None:
        super().__init__()
        self.__model = model
        self.__diff_vector = model.wv.get_vector(target_word) - model.wv.get_vector(source_word)

    def distance(self, source_word, target_word, metric=DistanceMetric.EUCLIDEAN):
        wv = self.__model.wv
        pair_diff_vector = wv.get_vector(target_word) - wv.get_vector(source_word)
        return self.__calculate_distance(self.__diff_vector, pair_diff_vector, metric)

    def are_similar(self, source_word, target_word):
        distance = self.distance(source_word, target_word, DistanceMetric.COSINE)
        return distance < self.__DISTANCE_THRESHOLD

    def get_distance(self, source_word, target_word):
        return self.distance(source_word, target_word, DistanceMetric.COSINE)

    def __calculate_distance(self, vector_a, vector_b, metric):
        return self.__DISTANCE_FUNCTIONS[metric](vector_a, vector_b)
