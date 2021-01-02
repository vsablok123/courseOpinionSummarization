# -*- coding: utf-8 -*-
#The idea is to save the Unigram features classes with all the members of that class.
"""
Created on Sun Dec  6 19:47:09 2020

@author: sablo
"""
import re
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import en_core_web_lg
nlp = en_core_web_lg.load()
from starterUtil import getCourseReviewsbyTag, getTagsList
from spellchecker import SpellChecker
from sklearn import cluster
from collections import OrderedDict 


K_FEATURES = 50
NUM_CLUSTERS = 400
courseReviews = getCourseReviewsbyTag()
topNounWords = {}
tags = getTagsList()
cleanDFwoStopwords = pd.read_csv("cleanReviewswithoutStopwords")
noOfCourses = 0

def getMinCountforNouns(noOfCourses):
    return (400 + (300*noOfCourses/613))

def createTopNounDict(tag):
    global topNounWords
    topNounWords = {}
    if tag == ' ':
        cleanedCorpus = cleanDFwoStopwords['reviews']
        noOfCourses = 613
    else:
        cleanedCorpus = cleanDFwoStopwords[cleanDFwoStopwords['Tags'] == tag]['reviews']
        noOfCourses = tags.loc[tag]
    noun_dict = {}
    nounTags = ['NN','NNS','NNP','NNPS']
    for review in cleanedCorpus:
        tokens = word_tokenize(str(review))
        postags = get_postags(tokens)
        for i in range(len(postags)):
            if postags[i] in nounTags:
                if tokens[i] in noun_dict:
                    noun_dict[tokens[i]] += 1
                else:
                    noun_dict[tokens[i]] = 1
    #Finding the most appearing nouns
    minCountforNouns = getMinCountforNouns(noOfCourses)
    for key,value in noun_dict.items():
        if value > minCountforNouns:
            topNounWords[key] = value
    print('Noun dictionary created for' + tag + '\n')
    return topNounWords
                    
def get_postags(row):
    
    postags = nltk.pos_tag(row)
    list_classes = list()
    for  word in postags:
        list_classes.append(word[1])
    
    return list_classes

    
def clusteredTopNouns(tag):
    topNounWords = createTopNounDict(tag)

    spell = SpellChecker(distance=1)  # set at initialization
    misspelled = spell.unknown(list(topNounWords.keys()))
    for word in misspelled:
        if spell.correction(word) in topNounWords and word in topNounWords:
            topNounWords[spell.correction(word)] = topNounWords[spell.correction(word)] + topNounWords.pop(word)
       
    wordvectors = {}
    for index,row in topNounWords.items():
        wordvector = nlp(index).vector
        wordvectors[index] = wordvector
        
    X = np.zeros((len(wordvectors),300))
    for i, (word, vector) in enumerate(wordvectors.items()):
        X[i] = vector

    kmeans = cluster.KMeans(n_clusters=int(len(X)/3),max_iter=1000)
    kmeans.fit(X)       
    labels = kmeans.labels_
    clusters = {}
    clsters = {}
    for (word,label) in zip([*wordvectors] , labels):

        count = topNounWords[word]
        if label in clusters:
            clusters[label].append((word, count))
        else:
            clusters[label] = [(word,count)]
            
    filteredFeatures = {}
    for clster in clusters:
        counter = list(zip(*clusters[clster]))[1]
        words = list(zip(*clusters[clster]))[0]
        word = words[counter.index(max(counter))]
        filteredFeatures[word] = max(counter)
        clsters[word] = words
    
    return filteredFeatures, clsters
        
    
    
def tagRelevantUnigramProb(unigrams, tag):
    streamReviews = courseReviews[courseReviews['Tags'] == tag]['reviews'].apply(lambda x: str(x))
    reviewText = " ".join(streamReviews)
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(reviewText)
    unigramsProb = OrderedDict()
    for word in unigrams:
        count = phraseCountinText(word,reviewText)
        if count != 0:
            unigramsProb[word] = -np.log(count/len(words))
        else:
            unigramsProb[word] = 100
    return unigramsProb  

def allUnigramsProb(unigrams):
    raw_text = " ".join(courseReviews['reviews'].apply(lambda x: str(x)))
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(raw_text)
    unigramsProb = OrderedDict()
    for word in unigrams:
        count = phraseCountinText(word,raw_text)
        unigramsProb[word] = -np.log(count/len(words))
    return unigramsProb        
            
def phraseCountinText(phrase, text):
    return len(re.findall(phrase,text))
            
def getTagRelevantFeatures(tag, unigrams, threshold):
    tagUnigramProb = tagRelevantUnigramProb(unigrams, tag)
    allUnigramProb = allUnigramsProb(unigrams)
    tagRelevantFeatures = {}
    for i, word in enumerate(tagUnigramProb):
        if allUnigramProb[word] - tagUnigramProb[word] > threshold:
            tagRelevantFeatures[word] = unigrams[word]
    return tagRelevantFeatures

unigrams, Clusters = clusteredTopNouns(' ')
tagFeaturesDict = {} 
def createFeatureslist():
    
    #Create General Nouns list (unigrams size = 400, counts size = 400)
    assocWordsinTagDict = dict()

    tagsList = ["['Arts and Humanities', 'Music and Art']", "['Business', 'Business Essentials']", 
                "['Business', 'Leadership and Management']","['Computer Science', 'Software Development']",
                "['Data Science', 'Data Analysis']", "['Data Science', 'Machine Learning']",
                "['Data Science', 'Probability and Statistics']","['Computer Science', 'Mobile and Web Development']",
                "['Health', 'Basic Science']", "['Personal Development', 'Personal Development']",
                "['Social Sciences', 'Governance and Society']","['Business', 'Business Strategy']",
                "['Business', 'Finance']","['Information Technology', 'Cloud Computing']"]                                                
    for tag in tagsList:
        print ("finding features for tag " + tag + '\n')
        generalCluster = dict()
        tagCluster = dict()
        tagUnigrams, clusters = clusteredTopNouns(tag)
        tagRelevantFeatures = getTagRelevantFeatures(tag, tagUnigrams, 0.8)
        top25general = {}
        #Picking top 25 tag specific features. 
        if len(tagRelevantFeatures) > 25:
            top25specific = sorted(tagRelevantFeatures,key = tagRelevantFeatures.get, reverse = True)[0:25]
        else :
            top25specific = tagRelevantFeatures
            
        #Creating final cluster for tag specific words.    
        for word in top25specific:
            tagCluster[word] = clusters[word]
            
        counter = 0  
        for word in sorted(unigrams, key = unigrams.get, reverse = True):
            if word in top25specific:
                continue
            else:
                top25general[word] = unigrams[word]
                generalCluster[word] = Clusters[word]
                counter = counter + 1
            if counter == 25:
                break
        
        #Merging the two dictionaries together
        assocWordsinTagDict = {**tagCluster, **generalCluster}
        global tagFeaturesDict
        tagFeaturesDict[tag] = assocWordsinTagDict
        
 
def finalMergedFeatures(tagFeaturesDict, Clusters, bigramfeatures):
    invalidFeatures = ['Andrew', 'Coursera', 'work', 'data', 'things', 'Great', 'time', 'Thank', 'way', 'lot', 'nan', 'course', 'life', 'understand']
    missedFeatures = ['instructor', 'basics', 'examples', 'topics', 'lectures', 'Professor', 'exercises', 'opportunity', 'project','theory','teacher']
    featurestoCombine = [['information','knowledge'],['class','lectures'],['instructor','Professor','teacher'],['assignments','exercises']]
    #removing the invalid features
    for tag, featureWords in tagFeaturesDict.items():
        newFeatureWords = featureWords.copy()
        for word,_ in featureWords.items():
            if word in invalidFeatures:
                newFeatureWords.pop(word)
        tagFeaturesDict[tag] = newFeatureWords
        
    #Adding the new features
    for tag, featureWords in tagFeaturesDict.items():
        for word in missedFeatures:
            if not (word in featureWords):
                featureWords[word] = Clusters[word]
                
    #Merging the unigram and bigram features
    for tag, featureWords in tagFeaturesDict.items():
        newFeatureWords = featureWords.copy()
        for word,_ in featureWords.items():
            for bigram in bigramfeatures[tag]:
                if word.lower() in bigram:
                    newFeatureWords.pop(word)
                    newFeatureWords[bigram] = [bigram]
                    break
        tagFeaturesDict[tag] = newFeatureWords
        
    #Combining similar meaning features
    featureWords = tagFeaturesDict["['Arts and Humanities', 'Music and Art']"]
    combinedfeatures = [sum(list(map(lambda x: featureWords[x] if x in featureWords else (), features)),()) for features in featurestoCombine]
    for tag, featureWords in tagFeaturesDict.items():
        for i, words in enumerate(featurestoCombine):
            for word in words: 
                if word in featureWords:
                    featureWords.pop(word) 
            featureWords[words[0]] = combinedfeatures[i]
            