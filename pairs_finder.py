from itertools import chain
from operator import itemgetter
from pprint import pprint
from typing import List

import fire
import gensim.downloader as api
import networkx as nx
from gensim.models.keyedvectors import Word2VecKeyedVectors
from networkx.drawing.nx_agraph import write_dot

from pairs_evaluator import PairsEvaluator

THE_MODEL = api.load('glove-wiki-gigaword-50')


class PairsFinder(object):
    __DEFAULT_MODEL_NAME = 'glove-wiki-gigaword-50'
    __SIMILARITY_THRESHOLD = 0.5

    def __init__(self) -> None:
        super().__init__()
        self.__model: Word2VecKeyedVectors = THE_MODEL

    def find(self, word1, word2):
        source_word = word1
        target_word = word2

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
                    g.add_edge(similar_source, best_target, weight=evaluator.get_distance(similar_source, best_target))

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

        nx.set_edge_attributes(g, nx.get_edge_attributes(g, 'weight'), 'label')
        g = g.subgraph(v for v, degree in g.degree if degree > 0)

        write_dot(g, 'my_dot.dot')
        pprint(list(g.edges))
        return list(g.edges)

    def __create_pairs_evaluator(self, source_word, target_word):
        return PairsEvaluator(self.__model, source_word, target_word)

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
