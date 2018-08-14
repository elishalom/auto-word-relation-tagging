from gensim.models.keyedvectors import Word2VecKeyedVectors
from scipy.spatial.distance import euclidean, cosine


class PairsEvaluator(object):
    __DISTANCE_THRESHOLD = 0.35

    def __init__(self, model: Word2VecKeyedVectors, source_word: str, target_word: str) -> None:
        super().__init__()
        self.__model = model
        self.__diff_vector = model.wv.get_vector(target_word) - model.wv.get_vector(source_word)

    def distance(self, source_word, target_word):
        wv = self.__model.wv
        pair_diff_vector = wv.get_vector(target_word) - wv.get_vector(source_word)
        return euclidean(self.__diff_vector, pair_diff_vector)

    def are_similar(self, source_word, target_word):
        distance = self.get_distance(source_word, target_word)
        return distance < self.__DISTANCE_THRESHOLD

    def get_distance(self, source_word, target_word):
        wv = self.__model.wv
        pair_diff_vector = wv.get_vector(target_word) - wv.get_vector(source_word)
        distance = cosine(self.__diff_vector, pair_diff_vector)
        return distance
