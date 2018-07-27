import nltk
import math
import operator
import time
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk import FreqDist
from elasticsearch import Elasticsearch
from collections import Counter
from functools import reduce
import pickle
import os.path

def queryElasticsearch(elaticsearch, term, num_results_per_query):
    """
    Query elasticsearch for the given term and return the query results
    
    Args:
        elaticsearch: the Elasticsearch instance to query
        term: the term to query for
        num_results_per_query: number of results to return
        
    Returns:
        The response of the query from elasticsearch 
    """
    query_template = """
                    {{
                      "query": {{
                        "query_string": {{
                          "default_field": "title",
                          "query": "{0}"
                        }}
                      }}
                    }}
                    """
    
    query_string = query_template.format(term)
    
    return elaticsearch.search(index="wikipedia", doc_type="doc", body=query_string, size=num_results_per_query)

def getDocuments(elaticsearch, term, num_results_per_query):
    """
    Query Elasticsearch for the given term and return sentences that
    contain all of the words in the term in the index from the top 'size' 
    documents in Elasticsearch
    
    Args:
        elaticsearch: the Elasticsearch instance to query
        term: the term to query for
        num_results_per_query: number of documents to query for
        
    Returns:
        List of sentences found that contain all the words
    """
    docs = queryElasticsearch(elaticsearch, term=term, num_results_per_query=num_results_per_query)['hits']['hits']
    result = []
    termTokens = term.split(" ")
    for doc in docs:
      try:
        sentences = list(filter(lambda x: len(x) > len(term), doc['_source']['text'][0].split(".")))
        for sentence in sentences:
          containsAll = True
          for termToken in termTokens:
            if termToken not in sentence:
              containsAll = False
              break
          if containsAll == True:
            result += [sentence]
      except:
        pass
    return result

def getNumHits(term, frequencies):
    """
    Get the frequency of the given term
    
    Args:
        term: the term to get the frequency of
        frequencies: a dictionary of terms and their frequencies
    
    Returns:
        The frequency of the term from the frequencies dict. 0 if it doesn't
        exist in th the dict
    """
    return frequencies.get(term, 0)

def filterWords(text, stop_words):
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

def tokenize(text, stop_words):
    """
    Tokenize the fiven text
    
    Args:
        text: the text to tokenize
        stop_words: a list of words to filter out
    Returns:
        text after tokenization, conversion to lowercase and filtering of words
        using the filterWords function
    """
    return filterWords(nltk.word_tokenize(text.lower()), stop_words)

def getTfIdf(term, count, totalDocuments, frequencies):
    """
    Calculate the tf/idf of the given term
    
    Args:
        term: the term to calculate tf/idf of
        count: the term frequency
        totalDocuments: the total number of documents
        frequencies: a dictionary that maps a word to its frequency in a corpus
        
    Returns:
        The tf/idf value for the given term
    """
    hitDocs = getNumHits(term, frequencies)
    if hitDocs == 0:
      return 0
    idf = math.log(totalDocuments / hitDocs)
    tf_idf = count * idf
    return tf_idf

def termCounter(elaticsearch, term, num_results_per_query, stop_words):
    """
    Count the occurrences of different words in the sentences that contain
    all the words of the given term
    
    Args:
        elaticsearch: the Elasticsearch instance to query
        term: the term to query for
        num_results_per_query: number of results returned per query
        stop_words: a list of words to filter out
        
    Returns:
        A counter of occurrences of all the unfiltered words that returned from
        the given query
    """
    res = getDocuments(elaticsearch, term, num_results_per_query)
    result_words = []
    for doc in res:
        result_words += tokenize(doc, stop_words)
    counter = Counter(result_words)
    return counter

def calculateMostProbableRelations(terms, topn=50, num_results_per_query=200):
    """
    Get the most probable labels for the given list of terms
    
    Args:
        terms: a list of terms to search for labels for
        topn: number of labels to return
        num_results_per_query: number of results returned per query
    Returns:
        A list of 'topn' labels which represent the relations between the words
        in the terms sorted in descending order of probablity
    """
    start = time.time()

    nltk.download('stopwords') 
    nltk.download('brown') 
    
    stop_words = stopwords.words('english') + ['category', 'also', 'list', 'align', 'url', 'title', 'center', 'thumb', 'date']
    
    elaticsearch = Elasticsearch()
    
    totalDocuments = elaticsearch.search(index="wikipedia", doc_type="doc", body={"query": {"match_all": {}}}, size=0)['hits']['total']
    
    frequenciesFileName = '../data/frequencies.pickle'
    
    try:
        print('Loading word frequencies dictionary...')
        frequencies = pickle.load(open(frequenciesFileName, 'rb'))
    except:
        print('Word frequencies dictionary doesn\'t exist; Creating it and saving it to file...')
        frequencies = FreqDist(word.lower() for word in brown.words())
        os.makedirs(os.path.dirname(frequenciesFileName), exist_ok=True)
        with open(frequenciesFileName, 'wb') as handle:
            pickle.dump(frequencies, handle, protocol = pickle.HIGHEST_PROTOCOL)
            
    print('Finished loading word frequencies dictionary')
    
    termsSet = set([word.lower() for term in terms for word in term.split()])
    
    termCounters = [termCounter(elaticsearch, x, num_results_per_query, stop_words) for x in terms]
    allTerms = reduce(lambda s1, s2: s1 | s2, map(lambda d: d.keys(), termCounters)) - termsSet
    allDict = {}

    for term in allTerms:
      product = 1
      for termc in termCounters:
        currentCount = termc.get(term)
        if(currentCount != None):
          product *= getTfIdf(term, currentCount, totalDocuments, frequencies)
      allDict[term] = product
    
    sortedResult = sorted(allDict.items(), key=operator.itemgetter(1), reverse=True)
    
    result = list(map(lambda x: x[0], list(filter(lambda pair: pair[1] > 0, sortedResult))[:topn]))
    print("elapsed: " + str(time.time() - start))
    
    return result


terms = ['Paris France', 'Dakar Senegal', 'Santiago Chile', 'Ottawa Canada', 'Moscow Russia', 'Jerusalem Israel', 'Vienna Austria', 'Kiev Ukraine']


terms = ['Austria city', 'Austria capital', 'Austria region', 'Austria region', 'Austria location', 'Austria national', 'Austria international', 'Austria country']

terms = ['Israel city', 'Israel capital', 'Israel region', 'Israel region', 'Israel location', 'Israel national', 'Israel international', 'Israel country']

terms = ['Jordan city', 'Jordan capital', 'Jordan region', 'Jordan region', 'Jordan location', 'Jordan national', 'Jordan international', 'Jordan country']

terms = ['Senegal city', 'Senegal capital', 'Senegal region', 'Senegal region', 'Senegal location', 'Senegal national', 'Senegal international', 'Senegal country']

#terms = ['Ukraine Chernobyl', 'Japan earthquake', 'USA tornado', 'Japan Fukushima']

#terms = ['USA Trump', 'Russia Putin', 'France Macron', 'Turkey Erdoğan']

#terms = ['Samsung Galaxy', 'Apple iPhone']

#terms = ['Google Android', 'Apple IOS', 'Microsoft Windows']

#terms = ['Germany Europe', 'Congo Africa', 'China Asia', 'Japan Asia', 'France Europe', 'Ethiopia Africa', 'India Asia', 'Chile America', 'Argentina America', 'Sudan Africa', 'Uganda Africa']

#terms = ['Owl Strigiformes', 'Giraffe Artiodactyla', 'Elephant Proboscidea', 'Dolphin Cetartiodactyla']

#terms = ['Owl Strigiformes', 'Elephant Proboscidea', 'Dolphin Cetartiodactyla']

#terms = ['Paris France', 'Jerusalem Israel']

#terms = ['King Queen', 'Actor Actress', 'waiter waitress', 'steward stewardess', 'husband wife', 'brother sister', 'mother father']

  
print(calculateMostProbableRelations(terms))
