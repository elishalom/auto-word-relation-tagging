import nltk
import math
import operator
import time
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
        """
        page_titles = []
        #res = requests.get("http://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srprop&srlimit=20&formatversion=2&srsearch=" + 
        #                   "|".join(term.split(" ")))
        #for item in res.json()['query']['search']:
        #    page_titles.append(item['title'])
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
        res = requests.get("https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exlimit=max&exintro&explaintext&formatversion=2&titles=" + 
                           "|".join(titles))
        result = []
        term_tokens = term.split(' ')
        for item in res.json()['query']['pages']:
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
            totalDocuments: the total number of documents
            frequencies: a dictionary that maps a word to its frequency in a corpus
            
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
            currentCount = termc.get(term)
            if(currentCount != None):
              product *= self.__get_tf_idf(term, currentCount)
          all_dict[term] = product
        
        sorted_result = sorted(all_dict.items(), key=operator.itemgetter(1), reverse=True)
        
        result = list(map(lambda x: x[0], list(filter(lambda pair: pair[1] > 0, sorted_result))[:topn]))
        
        return result
        
    
if __name__ == '__main__':    
    terms = [('Paris', 'France'), ('Dakar', 'Senegal'), ('Santiago', 'Chile'), ('Ottawa', 'Canada'), ('Moscow', 'Russia'), ('Jerusalem', 'Israel'), ('Vienna', 'Austria'), ('Kiev', 'Ukraine')]
    terms = [('seoul', 'korea'),('tehran', 'iran'),('warsaw', 'poland'),('tbilisi', 'armenia'),('belgrade', 'serbia'),('prague', 'bulgaria'),('bonn', 'eu'),('berlin', 'germany'),('bucharest', 'romania'),('damascus', 'syria'),('moscow', 'russia'),('minsk', 'belarus')]
    #terms = [('Seoul', 'Korea')]
    #terms = ['Austria capital', 'Austria city', 'Austria international', 'Austria population', 'Austria largest']
    #terms = ['Israel capital', 'Israel city', 'Israel international', 'Israel population', 'Israel largest']
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
    
    #terms = ['Paris France']
    producer = LabelProducer()
    labels = producer.calculate_most_probable_relations(terms)
    print(labels)