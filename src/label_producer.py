import nltk
import math
import operator
import pickle
import requests
import wikipedia
import os.path
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk import FreqDist
from collections import Counter
from functools import reduce


class LabelProducer(object):
    
    __TOTAL_ARTICLES = 5696797 # Number of articles in wikipedia. Used for tf-idf calculation
    __TERM_FREQUENCIES_DICT = {} # Term frequencies for words. Used for tf-idf calculation
    __EXTRACTS_PER_TERM = 20 # Number of document intros to process per search term pair
    
    
    def __init__(self) -> None:
        super().__init__()
    
    def __get_wiki_page_list(self, term):
        """
        Get a relevant page list for the given term
        
        Args:
            term: a string to use for querying wikipedia to get relevant documents
        
        Returns:
            A list of page titles related to the given term
        """
        page_titles = []
        page_titles = wikipedia.search(term)[:self.__EXTRACTS_PER_TERM]
        return page_titles
    
    def __get_pages(self, term):
        """
        Query Wikipedia for the given term and return sentences that
        contain all of the words in the term
        
        Args:
            term: the term to query for
            
        Returns:
            List of sentences found that contain all the words
        """
        titles = self.__get_wiki_page_list(term=term)
        result = []
        
        if len(titles) > 0:
            res = requests.get("https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exlimit=max&exintro&explaintext&formatversion=2&titles=" + 
                               "|".join(titles))
            term_tokens = term.split(' ')
            pages = res.json()['query']['pages']
            for item in pages:
              try:
                page = item['extract'].lower()
                sentences = list(filter(lambda x: len(x) > len(term), page.split(".")))
                for sentence in sentences:
                  contains_all = True
                  for term_token in term_tokens:
                    if term_token not in sentence:
                      contains_all = False
                      break
                  if contains_all == True:
                    result += [sentence]
              except Exception as e:
                print(e)
        return result
    
    def __get_term_frequency(self, term):
        """
        Get the frequency of the given term
        
        Args:
            term: the term to get the frequency of
        
        Returns:
            The frequency of the term from the frequencies dict. 0 if it doesn't
            exist in th the dict
        """
        return self.__TERM_FREQUENCIES_DICT.get(term, 0)
    
    def __get_tf_idf(self, term, count):
        """
        Calculate the tf/idf of the given term
        
        Args:
            term: the term to calculate tf/idf of
            count: the term frequency
            
        Returns:
            The tf/idf value for the given term
        """
        hit_docs = self.__get_term_frequency(term)
        if hit_docs == 0:
          return 0
        idf = math.log(self.__TOTAL_ARTICLES / hit_docs)
        tf_idf = count * idf
        return tf_idf

    def __filter_words(self, text, stop_words):
        """
        Filter out words from the given text
        
        Args:
            text: the text to filter
            stop_words: a list of words to filter out
        
        Returns:
            All the words in the text that are not in the stop_words list, have
            length greater than 1 and are alphabetical characters
        """
        return [word for word in text if word not in stop_words and len(word) > 1 and word.isalpha()]
    
    def __tokenize(self, text, stop_words):
        """
        Tokenize the fiven text
        
        Args:
            text: the text to tokenize
            stop_words: a list of words to filter out
            
        Returns:
            text after tokenization, conversion to lowercase and filtering of words
            using the filterWords function
        """
        return self.__filter_words(nltk.word_tokenize(text), stop_words)
    
    def __term_counter(self, term, stop_words):
        """
        Count the occurrences of different words in the sentences that contain
        all the words of the given term
        
        Args:
            term: the term to query for
            stop_words: a list of words to filter out
            
        Returns:
            A counter of occurrences of all the unfiltered words that returned from
            the given query
        """
        res = self.__get_pages(term)
        result_words = []
        for doc in res:
            result_words += self.__tokenize(doc.lower(), stop_words)
        counter = Counter(result_words)
        return counter
    
    def calculate_most_probable_relations(self, terms, topn=20):
        """
        Get the most probable labels for the given list of terms
        
        Args:
            terms: a list of 2-tuples which are terms to search labels for
            topn: number of labels to return
            
        Returns:
            A list of 'topn' labels which represent the relations between the words
            in the terms sorted in descending order of probablity
        """
    
        nltk.download('stopwords') 
        nltk.download('brown') 
        
        stop_words = stopwords.words('english')
        
        frequencies_file_name = '../data/frequency.pickle'

        term_str = list(map(lambda pair: pair[0] + " " + pair[1], terms))
        
        try:
            print('Loading word frequencies dictionary...')
            self.__TERM_FREQUENCIES_DICT = pickle.load(open(frequencies_file_name, 'rb'))
        except:
            print('Word frequencies dictionary doesn\'t exist; Creating it and saving it to file...')
            self.__TERM_FREQUENCIES_DICT = FreqDist(word.lower() for word in brown.words())
            os.makedirs(os.path.dirname(frequencies_file_name), exist_ok=True)
            with open(frequencies_file_name, 'wb') as handle:
                pickle.dump(self.__TERM_FREQUENCIES_DICT, handle, protocol = pickle.HIGHEST_PROTOCOL)
                
        print('Finished loading word frequencies dictionary')
        
        terms_set = set([word.lower() for term in term_str for word in term.split()])
        
        term_counters = [self.__term_counter(x, stop_words) for x in term_str]
        all_terms = reduce(lambda s1, s2: s1 | s2, map(lambda d: d.keys(), term_counters)) - terms_set
        all_dict = {}
    
        for term in all_terms:
          product = 1
          for termc in term_counters:
            current_count = termc.get(term)
            if(current_count != None):
              product *= self.__get_tf_idf(term, current_count)
          all_dict[term] = product
        
        sorted_result = sorted(all_dict.items(), key=operator.itemgetter(1), reverse=True)
        
        result = list(map(lambda x: x[0], list(filter(lambda pair: pair[1] > 0, sorted_result))[:topn]))
        
        return result