# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
"""
Created on Sat Dec  5 20:15:14 2020

@author: sablo
"""
def extractLettersFromReview(x):
    return(''.join(re.sub('[^a-zA-Z_.?!,;]', ' ', x)))

def extractSentencesFromReview(x):
    pattern = '(?<!\.[a-zA-Z])[\.\?\!](?![a-zA-Z]\.)'
    return re.split(pattern, x)

def cleanReview(x):
    sentences = extractSentencesFromReview(x)
    y = sentences[0]
    for sentence in sentences[1:]:
        if len(sentence) > 0 and sentence[0] != ' ':
            y = y + '. ' + sentence
        else:
            y = y + '.' + sentence
        print(y)
    y = extractLettersFromReview(y)
    return y  

def isinEnglish(x):
    stopwrds = stopwords.words("english")
    count = 0
    words = str(x).split(' ')
    for word in words:
        if word in stopwrds:
            count = count + 1
        if count == 3:
            return True
    return False

#CourseReviewDF - Keys -> Tags, Values -> Course_id, Reviews
#Returns corpus text for that tag
def generateCleanDF(CourseReviewDF):

    cleanDF = CourseReviewDF[CourseReviewDF.reviews.apply(lambda x : isinEnglish(x))]['reviews'].apply(lambda y: cleanReview(y)).copy()
    del CourseReviewDF
    return cleanDF
    
def removeStopwords(cleanDF):
    newCorpus = []
    stopwrds = stopwords.words("english")
    corpus = cleanDF['reviews']
    for review in corpus:
       words_clean = []
       tokens = word_tokenize(review)
       for word in tokens:
        if word not in stopwrds:
            words_clean.append(word)
       review = ' '.join(words_clean)
       newCorpus.append(review)
    cleanDF['reviews'] = newCorpus
    return cleanDF
