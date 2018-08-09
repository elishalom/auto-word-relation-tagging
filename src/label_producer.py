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

class LabelProducer(object):
    
    def __init__(self) -> None:
        super().__init__()
    
    def __queryElasticsearch(self, elaticsearch, term, num_results_per_query):
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
    
    def __getDocuments(self, elaticsearch, term, num_results_per_query):
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
        docs = self.__queryElasticsearch(elaticsearch, term=term, num_results_per_query=num_results_per_query)['hits']['hits']
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
    
    def __getNumHits(self, term, frequencies):
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
    
    def __filterWords(self, text, stop_words):
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
        return self.__filterWords(nltk.word_tokenize(text.lower()), stop_words)
    
    def __getTfIdf(self, term, count, totalDocuments, frequencies):
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
        hitDocs = self.__getNumHits(term, frequencies)
        if hitDocs == 0:
          return 0
        idf = math.log(totalDocuments / hitDocs)
        tf_idf = count * idf
        return tf_idf
    
    def __termCounter(self, elaticsearch, term, num_results_per_query, stop_words):
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
        res = self.__getDocuments(elaticsearch, term, num_results_per_query)
        result_words = []
        for doc in res:
            result_words += self.__tokenize(doc, stop_words)
        counter = Counter(result_words)
        return counter
    
    def calculateMostProbableRelations(self, terms, topn=50, num_results_per_query=200):
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
        #start = time.time()
    
        nltk.download('stopwords') 
        nltk.download('brown') 
        
        stop_words = stopwords.words('english') + ['category', 'also', 'list', 'align', 'url', 'title', 'center', 'thumb', 'date', 'br']
        
        elaticsearch = Elasticsearch()
        
        totalDocuments = elaticsearch.search(index="wikipedia", doc_type="doc", body={"query": {"match_all": {}}}, size=0)['hits']['total']
        
        #frequenciesFileName = '../data/frequency.pickle'
        frequenciesFileName = '../data/wordCounts.pickle'
        
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
        
        termCounters = [self.__termCounter(elaticsearch, x, num_results_per_query, stop_words) for x in terms]
        allTerms = reduce(lambda s1, s2: s1 | s2, map(lambda d: d.keys(), termCounters)) - termsSet
        allDict = {}
    
        for term in allTerms:
          product = 1
          for termc in termCounters:
            currentCount = termc.get(term)
            if(currentCount != None):
              product *= self.__getTfIdf(term, currentCount, totalDocuments, frequencies)
          allDict[term] = product
        
        sortedResult = sorted(allDict.items(), key=operator.itemgetter(1), reverse=True)
        
        result = list(map(lambda x: x[0], list(filter(lambda pair: pair[1] > 0, sortedResult))[:topn]))
        #print("elapsed: " + str(time.time() - start))
        
        return result
    
    
    #terms = ['Paris France', 'Dakar Senegal', 'Santiago Chile', 'Ottawa Canada', 'Moscow Russia', 'Jerusalem Israel', 'Vienna Austria', 'Kiev Ukraine']
    
    
    #terms = ['Austria city', 'Austria capital', 'Austria region', 'Austria region', 'Austria location', 'Austria national', 'Austria international', 'Austria country']
    
    #terms = ['Israel city', 'Israel capital', 'Israel region', 'Israel region', 'Israel location', 'Israel national', 'Israel international', 'Israel country']
    
    #terms = ['Jordan city', 'Jordan capital', 'Jordan region', 'Jordan region', 'Jordan location', 'Jordan national', 'Jordan international', 'Jordan country']
    
    #terms = ['Senegal city', 'Senegal capital', 'Senegal region', 'Senegal region', 'Senegal location', 'Senegal national', 'Senegal international', 'Senegal country']
    
    #terms = ['Ukraine Chernobyl', 'Japan earthquake', 'USA tornado', 'Japan Fukushima']
    
    #terms = ['USA Trump', 'Russia Putin', 'France Macron', 'Turkey Erdoğan']
    
    #terms = ['Samsung Galaxy', 'Apple iPhone']
    
    #terms = ['Google Android', 'Apple IOS', 'Microsoft Windows']
    
    #terms = ['Germany Europe', 'Congo Africa', 'China Asia', 'Japan Asia', 'France Europe', 'Ethiopia Africa', 'India Asia', 'Chile America', 'Argentina America', 'Sudan Africa', 'Uganda Africa']
    
    #terms = ['Owl Strigiformes', 'Giraffe Artiodactyla', 'Elephant Proboscidea', 'Dolphin Cetartiodactyla']
    
    #terms = ['Owl Strigiformes', 'Elephant Proboscidea', 'Dolphin Cetartiodactyla']
    
    #terms = ['Paris France', 'Jerusalem Israel']
    
    #terms = ['King Queen', 'Actor Actress', 'waiter waitress', 'steward stewardess', 'husband wife', 'brother sister', 'mother father']
    
    #terms = ['shark dove','shark peacock','shark sky','dolphin liberty','dolphin neon','dolphin dove','whale dove','whale floats','whale craft','crocodile banner','crocodile dove','crocodile maroon','alligator maroon','alligator banner','alligator multicolored','bird banner','bird convention','bird flag','cat banner','cat neon','cat dove','squid floats','squid lanterns','squid skimmer','octopus lanterns','octopus floats','octopus orb','rat stag','rat flaming','rat lantern','sharks maple','sharks devils','sharks calgary','whales doves','whales floats','whales balloons','snake heavenly','bites floats','bites pancake','bites banner','birds flags','birds coloured','birds brightly','bite roll','bite shake','bite banner','reptile beechwood','reptile statuary','reptile cypress','fish green','fish colored','fish sugar','rabbit stag','rabbit banner','owl neon','owl maroon','owl cherokee','crocodiles lanterns','crocodiles worshipping','crocodiles heavenly','saltwater calabash','saltwater dunes','saltwater manitou','turtles lanterns','turtles symbolizing','frog lantern','frog maroon','snakes lanterns','snakes floats','snakes heavenly','wild black','wild parade','wild banner','dog banner','elephant banner','elephant symbolizing','elephant stag','cats maroon','cats bonfires','cats neon','boar stag','boar festooned','boar symbolizing','mosquito camel','mosquito coolers','mosquito caravan','bitten paraded','bitten whips','bitten leaned','droppings candies','droppings torches','droppings menorahs','jellyfish lanterns','jellyfish glowed','jellyfish orbs','insect symbolize','insect symbolizing','insect tulip','ant stag','ant lantern','ant tavern','finning cerulean','finning m&m','finning powerlink','dogs flags','dogs marchers','dogs parade','lizard maroon','lizard neon','spider floats','spider heavenly','spider lantern','prey floats','prey lanterns','prey symbolize','fishes floats','fishes heavenly','fishes lanterns','duck floats','pelican maroon','pelican fairport','pelican eureka','burrowing canopies','burrowing unfurl','burrowing inducting','sandbar fronting','sandbar colonnade','sandbar veranda','rodent stag','rodent totem','rodent tulip','ducks maple','ducks flags','ducks decked','animal umbrella','animal spirit','animal convention','thresher m&m','thresher wahine','thresher boneyard','parrot tricolour','parrot maroon','squirrel cherokee','squirrel maroon','hyena maroon','hyena cilla','hyena crowes','predator nomad','predator thunderbolt','predator patriot','blacktip lorikeet','blacktip ashmore','blacktip inhumans','ape heavenly','ape stag','ape symbolizing','gulper yoruban','gulper moonglows','gulper bayanihan','tick bandwagon','tick mlk','tick floats','spotted flags','spotted waving','spotted flag','panda symbolizing','panda inaugurate','panda jubilee','monkey neon','monkey symbolizing','pig banner','pig candy','pig slogan','mammal totem','mammal audubon','mammal enshrined','mouth onto','mouth forth','mouth banner','predatory vested','predatory umbrella','predatory wiccan','nest symbolizing','nest menorahs','orca lorikeet','orca rambler','orca boneyard','humpback shania','mako lorikeet','mako nereid','mako klf','kangaroo maroon','kangaroo barkers','wolf banner','raccoon maroon','raccoon sitka','raccoon cherokee','crabs lanterns','crabs heavenly','crabs floats','pigs marchers','pigs banner','pigs torches','hunters pioneers','hunters cherokee','hunters flags','bug neon','bug lights','bug banner','cookiecutter futureheads','cookiecutter front-man','cookiecutter opensocial','beast banner','beast patriotic','abalone s-70','abalone maroon','abalone multicolored','animals worship','animals spirit','animals symbolize','creature heavenly','creature iconic','creature shining','mole sherbet','mole maroon','mole heavenly','mammals faiths','venomous oppositions','venomous espousing','marsupial lorikeet','marsupial nomad','marsupial totem','reptiles creeds','reptiles lanterns','reptiles heavenly','caged bonfires','caged candle','caged symbolizing','killer icon','killer jesse','killer slain','insects lanterns','insects masses','leopard maroon','leopard logo','leopard emblazoned','fowl caravans','fowl zulus','fowl kwanzaa','hammerhead lorikeet','hammerhead samata','hammerhead multicoloured','rats lit','rats assemblies','rats floats','endangered liberty','endangered convention','endangered enshrined','crab pancake','crab blueberry','alligators flagpoles','alligators seashores','alligators lanterns','osterman cuffe','osterman woodie','osterman lorikeet','cow candy','cow candle','cow banner','albatross furled','albatross lorikeet']
    
    #terms = ['moscow ukraine','moscow russian','moscow belarus','kiev ukraine','kiev belarus','kiev bulgaria','prague bulgaria','prague romania','prague poland','warsaw poland','warsaw hungary','warsaw ukraine','belgrade serbia','belgrade yugoslavia','belgrade croatia','berlin germany','berlin austria','berlin denmark','minsk belarus','minsk ukraine','minsk moldova','tehran iran','tehran libya','vienna austria','vienna germany','vienna switzerland','tbilisi armenia','tbilisi belarus','tbilisi tajikistan','bonn eu','bonn germany','petersburg ukraine','petersburg spain','seoul korea','seoul china','seoul taiwan','damascus syria','damascus egypt','damascus iran','bucharest romania','bucharest bulgaria','bucharest hungary','athens greece','athens hungary','athens bulgaria','brussels eu','brussels belgium','helsinki finland','helsinki kazakhstan','helsinki estonia','budapest hungary','budapest austria','budapest bulgaria','beijing china','beijing taiwan','pyongyang korea','embassy u.s.','munich germany','munich switzerland','munich austria','zagreb croatia','zagreb yugoslavia','zagreb serbia','vladivostok kazakhstan','stockholm sweden','stockholm austria','stockholm denmark','ankara turkey','ankara ethiopia','paris france','paris french','paris belgium','hamburg germany','hamburg switzerland','hamburg austria','cairo egypt','cairo arabia','cairo morocco','istanbul turkey','istanbul arabia','istanbul denmark','tokyo japan','tokyo japanese','tokyo china','diplomats countries','washington states','beirut lebanon','beirut egypt','beirut algeria','visit china','tel israel','vilnius lithuania','vilnius estonia','vilnius bulgaria','riga lithuania','riga latvia','riga estonia','serbian republic','rome italy','rome spain','rome portugal','aviv israel','sofia bulgaria','sofia romania','sofia hungary','central states']
      
    
    #terms =['france prohertrib','french parisian','belgium stockholm','spain aires','netherlands amsterdam','italy rome','germany berlin','european london','switzerland zurich','europe opened','belgian cafe','portugal buenos','luxembourg budapest','austria vienna','eu brussels','denmark copenhagen','united held','brazil janeiro','canada vancouver','republic prague','russia moscow','nations geneva','argentina lima','chirac elysee','africa johannesburg','union offices','sweden gothenburg','world exhibition','italians parisians','advance closing','states d.c.','croatia zagreb','euro closes']
    #print(calculateMostProbableRelations(terms))
