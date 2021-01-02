# -*- coding: utf-8 -*-
#The idea is to save the Bigram features classes with all the members of that class.

"""
Created on Sun Dec  6 20:58:44 2020

@author: sablo
"""
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import RegexpTokenizer
from starterUtil import getCourseReviewsbyTag, getTagsList
from preprocessing_Util import generateCleanDF, removeStopwords
import spacy
import en_core_web_lg
from spacy import displacy
import pandas as pd
nlp = en_core_web_lg.load()
import re
import numpy as np

#courseReviews = getCourseReviewsbyTag()
#tags = getTagsList()
cleanedCorpus = pd.Series()
cleanDFwoStopwords = pd.read_csv("cleanReviewswithoutStopwords")
tags = getTagsList()
noOfCourses = 0
tagbigramProb = {}
allbigramProb = {}

def get_bigram_likelihood(statements, tag, freq_filter=3, nbest=200):
    """
    Returns n (likelihood ratio) bi-grams from a group of documents
    :param        statements: list of strings
    :param output_file: output path for saved file
    :param freq_filter: filter for # of appearances in bi-gram
    :param       nbest: likelihood ratio for bi-grams
    """

    #words = list()
    #tokenize sentence into words
    #for statement in statements:
        # remove non-words
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(statements)
    print(freq_filter)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigram_finder = BigramCollocationFinder.from_words(words)

    # only bi-grams that appear n+ times
    bigram_finder.apply_freq_filter(freq_filter)

    # TODO: use custom stop words
    bigram_finder.apply_word_filter(lambda w: len(w) < 3)

    bigram_results = bigram_finder.nbest(bigram_measures.likelihood_ratio, nbest)
    if tag == ' ':
        num = 300
    else:
        num = 50
    return bigram_finder.score_ngrams(bigram_measures.likelihood_ratio)[0:num]

def check_collocation_type(doc, collocation):
    index = []
    length =len(doc)
    for i in range(length-1):
        if collocation.split(' ')[0] in [doc[i].text, doc[i].lemma_]:
            if collocation.split(' ')[1] in [doc[i+1].text, doc[i+1].lemma_]:
                index.append(i)
    if len(index)> 0:
        return [(doc[index[0]].text, doc[index[0]+1].text),(doc[index[0]].pos_, doc[index[0]+1].pos_)]
    else:
        return False

def select_sentence(review, collocation):
    sentence_list = review.split('.')
    selected_sentences = [sentence for sentence in sentence_list if collocation in sentence]
    
    return ' '.join(selected_sentences)
    
def gen(n):
    while True:
        yield n
        n += 1
        
def streamRelevantBigramProb(bigrams, raw_text):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(raw_text)
    bigramsProb = {}
    for (word1, word2), _ in bigrams:
        phrase = word1 + " " + word2
        count = phraseCountinText(phrase,raw_text)
        if count != 0:
            bigramsProb[phrase] = -np.log(count/(len(words) + 1))
    return bigramsProb  

def allBigramsProb(bigrams, raw_text):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(raw_text)
    bigramsProb = {}
    for (word1, word2), _ in bigrams:
        phrase = word1 + " " + word2
        count = phraseCountinText(phrase,raw_text)
        bigramsProb[phrase] = -np.log(count/(len(words)+1))
    return bigramsProb

def phraseCountinText(phrase, text):
    return len(re.findall(phrase,text))

def getBigrams(tag):
    if tag == ' ':
        cleanedCorpus = cleanDFwoStopwords['reviews'].apply(lambda x: str(x))
        nbest = 300
        noOfCourses = 613
    else:
        cleanedCorpus = cleanDFwoStopwords[cleanDFwoStopwords['Tags'] == tag]['reviews'].apply(lambda x: str(x))
        nbest = 100
        noOfCourses = tags.loc[tag]

    rawtext = ' '.join(cleanedCorpus)
    #review_list = cleanedCorpus.tolist()
    bigrams = get_bigram_likelihood(rawtext.lower(), tag , freq_filter=getMinFreqforBigrams(noOfCourses), nbest = nbest)
    #bigram_joint = [' '.join(list(s[0])) for s in bigrams]
    
#    selected_collocations_with_type = []
#    for collocation in bigram_joint:
#
#        selected_data = [select_sentence(review.lower(), collocation) for review in review_list if collocation in review]
#
#    
#        g = gen(0)
#        while True:
#            index = next(g)
#            if index >= len(selected_data):
#                break
#            review = selected_data[index]
#            doc= nlp(review)
#            check = check_collocation_type(doc, collocation)
#            if check != False:
#                if check[1][0] == 'NOUN' and check[1][1] == 'NOUN':
#                    selected_collocations_with_type.append(check)       
#                    full_data_doc = [(token, token.ent_type_, token.pos_, token.lemma_) for token in doc]
#                    break
#            else:   
#                continue
    print("Bigrams found for tag " + tag + '\n')
    if tag == ' ':
        global allbigramProb
        allbigramProb = allBigramsProb(bigrams, rawtext)
    else:
        global tagbigramProb
        tagbigramProb = streamRelevantBigramProb(bigrams, rawtext)
    
    return bigrams
        
  
def getMinFreqforBigrams(noOfCourses):
    return (50 + (500*noOfCourses/613))


def getTagRelevantFeatures(tag, bigrams, threshold):

    tagRelevantFeatures = []
    for i, word in enumerate(tagbigramProb):
        if (not word in allbigramProb) or allbigramProb[word] - tagbigramProb[word] > threshold:
            tagRelevantFeatures.append(word)
    return tagRelevantFeatures
    
def createFeaturesList():
    
    #Create General Nouns list 
    bigrams = getBigrams(' ')
    tagFeaturesDict = {}

    tagsList = list(tags.index.values)
    for tag in tagsList:
        tagBigrams = getBigrams(tag)
        tagRelevantFeatures = getTagRelevantFeatures(tag, tagBigrams, 0.8)
        top10general = []
        #Picking top 10 tag specific features. 
        if len(tagRelevantFeatures) > 10:
            top10specific = tagRelevantFeatures[0:10]
        else :
            top10specific = tagRelevantFeatures
        counter = 0    
        for word in allbigramProb:
            if word in top10specific:
                continue
            else:
                top10general.append(word)
                counter = counter + 1
            if counter == 10:
                break
        
        tagFeaturesDict[tag] = top10specific + top10general
    return tagFeaturesDict    
            
       
