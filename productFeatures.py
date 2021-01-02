# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 19:47:09 2020

@author: sablo
"""
import re
import itertools
import pandas as pd
import datetime as dt
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import RegexpTokenizer
import string
import scipy.sparse as sparse
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from nltk.tokenize import word_tokenize

import spacy
import en_core_web_lg
from spacy import displacy
nlp = en_core_web_lg.load()
from starterUtil import getCourseReviewsbyTag
from preprocessing_Util import generateCorpusbyTag, removeStopwords

def createTopNounDict(tag):
    courseReviews = getCourseReviewsbyTag()
    if tag != ' ':
        courseReviews = courseReviews[courseReviews['Tags'] == tag]
        
    #Cleaning the reviews
    corpus = generateCorpusbyTag(courseReviews, tag)
    cleanedCorpus = removeStopwords(corpus)
    noun_dict = {}
    nounTags = ['NN','NNS','NNP','NNPS']
    for review in cleanedCorpus:
        tokens = word_tokenize(review)
        postags = get_postags(tokens)
        for i in range(len(postags)):
            if postags[i] in nounTags:
                if tokens[i] in noun_dict:
                    noun_dict[tokens[i]] += 1
                else:
                    noun_dict[tokens[i]] = 1
    #Finding the most appearing nouns
    topNounWords = {}
    for key,value in noun_dict.items():
        if value > 700:
            topNounWords[key] = value
    return topNounWords
                    
def get_postags(row):
    
    postags = nltk.pos_tag(row)
    list_classes = list()
    for  word in postags:
        list_classes.append(word[1])
    
    return list_classes



