from itertools import chain
from operator import itemgetter
from typing import List

import fire
import gensim.downloader as api
import networkx as nx
from gensim.models.keyedvectors import Word2VecKeyedVectors

from distance_metric import DistanceMetric
from label_producer import LabelProducer
from pairs_evaluator import PairsEvaluator


class PairsFinder(object):
    __DEFAULT_MODEL_NAME = 'glove-wiki-gigaword-50'
    __SIMILARITY_THRESHOLD = 0.5

    def __init__(self) -> None:
        super().__init__()
        self.__model: Word2VecKeyedVectors = api.load(self.__DEFAULT_MODEL_NAME)

    def find(self, word1, word2):
        source_word = word1.lower()
        target_word = word2.lower()

        try:
            evaluator = self.__create_pairs_evaluator(source_word, target_word)

            similar_sources = self.__extract_similar_sources(source_word)
            g = nx.DiGraph()
            g.add_weighted_edges_from([(source_word, target_word, 0)])
            for similar_source in similar_sources:
                similar_targets = self.__find_similar_targets(similar_source, source_word, target_word)

                best_targets = sorted(similar_targets,
                                      key=lambda similar_target: evaluator.distance(similar_source, similar_target))[:1]
                for best_target in best_targets:
                    if evaluator.are_similar(similar_source, best_target):
                        g.add_edge(similar_source, best_target, weight=evaluator.distance(similar_source, best_target,
                                                                                          metric=DistanceMetric.COSINE))

            terms = self.__resolve_ambiguities(g)

            producer = self.__create_label_producer()

            labels = producer.calculate_most_probable_relations(terms)
        except:
            labels = []

        return labels

    @staticmethod
    def __resolve_ambiguities(g):
        g: nx.DiGraph = g.subgraph(v for v, degree in g.degree if degree <= 4).copy()
        while any(degree > 1 for _, degree in g.degree):
            undecided_nodes = [v for v, degree in g.degree if degree > 1]
            contesting_edges = chain(g.in_edges(undecided_nodes, data='weight'),
                                     g.out_edges(undecided_nodes, data='weight'))
            source, target, _ = min(contesting_edges, key=itemgetter(2))
            g.remove_edges_from([(s, t) for s, t in g.out_edges(source) if t != target])
            g.remove_edges_from(list(g.in_edges(source)))
            g.remove_edges_from([(s, t) for s, t in g.in_edges(target) if s != source])
            g.remove_edges_from(list(g.out_edges(source)))
        g = g.subgraph(v for v, degree in g.degree if degree > 0)
        terms = list(g.edges)
        return terms

    def analogy(self, source_A, target_A, source_B, topn=5):
        labels = self.find(source_A, target_A)
        terms = []
        for i in range(topn):
            terms.append(labels[i] + " " + source_B)
            
        producer = self.__create_label_producer()
        new_labels = producer.calculate_most_probable_relations(terms)
        
        return new_labels

    def __create_pairs_evaluator(self, source_word, target_word):
        return PairsEvaluator(self.__model, source_word, target_word)

    @staticmethod
    def __create_label_producer():
        return LabelProducer()

    def __find_similar_targets(self, similar_source: str, source_word: str,
                               target_word: str, num_of_words: int = 20) -> List[str]:
        return list(map(itemgetter(0),
                        self.__model.most_similar([similar_source, target_word], [source_word],
                                                  topn=num_of_words)))

    def __extract_similar_sources(self, source_word: str, num_of_words: int = 20) -> List[str]:
        most_similar = self.__model.most_similar(source_word, topn=num_of_words)
        return list(map(itemgetter(0), filter(lambda p: p[1] > self.__SIMILARITY_THRESHOLD, most_similar)))


if __name__ == '__main__':
    fire.Fire(PairsFinder)
