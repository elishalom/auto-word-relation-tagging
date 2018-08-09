from operator import itemgetter
from typing import List

import fire
import gensim.downloader as api
from gensim.models.keyedvectors import Word2VecKeyedVectors

from pairs_evaluator import PairsEvaluator
from label_producer import LabelProducer


class PairsFinder(object):
    __DEFAULT_MODEL_NAME = 'glove-wiki-gigaword-50'
    __SIMILARITY_THRESHOLD = 0.6

    def __init__(self) -> None:
        super().__init__()
        self.__model: Word2VecKeyedVectors = api.load(self.__DEFAULT_MODEL_NAME)

    def find(self, word1, word2):
        source_word = word1
        target_word = word2

        evaluator = self.__create_pairs_evaluator(source_word, target_word)

        similar_sources = self.__extract_similar_sources(source_word)
        similar_targets = self.__extract_similar_sources(target_word)
        all_words = set()
        
        terms = []
        
        for similar_source in similar_sources:
            similar_targets = self.__find_similar_targets(similar_source, source_word, target_word)

            best_targets = sorted(similar_targets,
                                  key=lambda similar_target: evaluator.distance(similar_source, similar_target))[:3]
            for best_target in best_targets:
                if evaluator.are_similar(similar_source, best_target) and similar_source not in all_words and best_target not in all_words:
                    #print(similar_source, best_target, evaluator.distance(similar_source, best_target))
                    terms.append(similar_source + " " + best_target)
                    all_words.add(similar_source)
                    all_words.add(best_target)
            
        producer = self.__create_label_producer()
        
        labels = producer.calculateMostProbableRelations(terms)
        
        print(labels)

    def __create_pairs_evaluator(self, source_word, target_word):
        return PairsEvaluator(self.__model, source_word, target_word)
    
    def __create_label_producer(self):
        return LabelProducer()

    def __find_similar_targets(self, similar_source: str, source_word: str,
                               target_word: str, num_of_words: int = 20) -> List[str]:
        return list(map(itemgetter(0),
                        self.__model.most_similar([similar_source, target_word], [source_word],
                                                  topn=num_of_words)))

    def __extract_similar_sources(self, source_word: str, num_of_words: int = 100) -> List[str]:
        most_similar = self.__model.most_similar(source_word, topn=num_of_words)
        return [source_word] + list(
            map(itemgetter(0), filter(lambda p: p[1] > self.__SIMILARITY_THRESHOLD, most_similar)))


if __name__ == '__main__':
    fire.Fire(PairsFinder)
