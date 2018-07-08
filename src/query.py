import nltk
import math
import operator
import time
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
from collections import Counter
from functools import reduce

nltk.download('punkt')
nltk.download('stopwords') 

num_results_per_query = 100
topn = 100

stop_words = stopwords.words('english') + ['br', '\\/', 'category']

es = Elasticsearch()

totalDocuments = es.search(index="wikipedia", doc_type="doc", body={"query": {"match_all": {}}}, size=0)['hits']['total']

def queryElasticsearch(term, size):
  query_template = """
                {{
                  "query": {{
                    "query_string": {{
                      "default_field": "_all",
                      "query": "{0}"
                    }}
                  }}
                }}
                """

  query_string = query_template.format(term)

  return es.search(index="wikipedia", doc_type="doc", body=query_string, size=size)

def getDocuments(term, num):
  docs = sorted(queryElasticsearch(term=term, size=num * 10)['hits']['hits'], key=lambda hit: len(hit['_source']['text'][0]), reverse=True)[:num]
  return docs

def getNumHits(term):
  return queryElasticsearch(term, 0)['hits']['total']

def filterWords(text):
  return [word for word in text if word not in stop_words and len(word) > 1 and word.isalpha()]

def tokenize(text):
  return filterWords(nltk.word_tokenize(text.lower()))

def getTfIdf(term, count):
  hitDocs = getNumHits(term)
  if hitDocs == 0:
    return 0
  idf = math.log(totalDocuments / hitDocs)
  tf_idf = count * idf
  return tf_idf

def termCounter(term, items=num_results_per_query):
  res = getDocuments(term, items)
  result_words = []
  for doc in res:
    result_words += tokenize(doc['_source']['text'][0])

  return Counter(result_words)

def tfIdfDict(counter):
  tdIdfDict = {}
  for (term,count) in counter.items():
    tdIdfDict[term] = getTfIdf(term, count)

  return tdIdfDict

def tfIdfProductDict(dict1, dict2):
  productDict = {}
  for (term, tfIdf1) in dict1.items():
    tfIdf2 = dict2.get(term)
    if tfIdf2 != None:
      productDict[term] = tfIdf2 * tfIdf1

  return productDict

def getTfIdfDictByTerm(term):
  c = termCounter(term)
  return tfIdfDict(c)

def aggregatedTfIdfProduct(terms):
  productDict = getTfIdfDictByTerm(terms[0])

  for term in terms[1:]:
    newDict = {}
    counter = termCounter(term)
    for (term,count) in counter.items():
      tfIdfPrev = productDict.get(term)
      if tfIdfPrev != None:
        newDict[term] = tfIdfPrev * getTfIdf(term, count)
      #else:
      #  newDict[term] = getTfIdf(term, count)
    productDict = newDict

  return productDict


start = time.time()

#d1 = getTfIdfDictByTerm('Paris France')
#d2 = getTfIdfDictByTerm('London England')
#d3 = getTfIdfDictByTerm('Dakar Senegal')
#d4 = getTfIdfDictByTerm('Wellington New Zealand')
#d5 = getTfIdfDictByTerm('Santiago Chile')
#d6 = getTfIdfDictByTerm('Ottawa Canada')
#productDict = reduce((lambda x, y: tfIdfProductDict(x, y)), [d1, d2, d3, d4, d5, d6])

terms = ['Paris France', 'London United Kingdom', 'Dakar Senegal', 'Wellington New Zealand', 'Santiago Chile', 'Ottawa Canada', 
  'Moscow Russia', 'Jerusalem Israel', 'Prague Czech Republic']

#terms = ['Paris France', 'Jerusalem Israel']

#terms = ['King Queen', 'Actor Actress', 'waiter waitress', 'steward stewardess', 'husband wife', 'brother sister', 'mother father']

termCounters = [termCounter(x) for x in terms]

allTerms = reduce(lambda s1, s2: s1 | s2, map(lambda d: d.keys(), termCounters))

allDict = {}

for term in allTerms:
  appearances = 0
  product = 1
  for termCounter in termCounters:
    currentCount = termCounter.get(term)
    if(currentCount != None):
      appearances += 1
      product *= getTfIdf(term, currentCount)
  if appearances >= len(termCounters):
    allDict[term] = product


sortedResult = sorted(allDict.items(), key=operator.itemgetter(1), reverse=True)
#productDict = aggregatedTfIdfProduct(terms)

#sortedResult = sorted(productDict.items(), key=operator.itemgetter(1), reverse=True)
i = 0
for (term, tfIdfProduct) in sortedResult:
  print(term + ", " + str(tfIdfProduct))
  i += 1
  if i == topn:
    break

print("elapsed: " + str(time.time() - start))

#print(Counter(result_words))
#print(totalDocuments)

#print(res['hits']['hits'][0]['_source']['text'][0])
#print([word for word in text if word not in stopwords.words('english')])