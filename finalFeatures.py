# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:33:23 2020

@author: sablo
"""
from unigramFeatures import Clusters

import pickle 
   
try: 
    finalFeatures = open('finalFeatures', 'wb') 
    pickle.dump(tagFeaturesDict, finalFeatures) 
    finalFeatures.close() 
  
except: 
    print("Something went wrong")

# open a file, where you stored the pickled data
file = open('finalFeatures', 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()

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
                else:
                    newFeatureWords[bigram] = [bigram]
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
            
            
bigramfeatures = {"['Arts and Humanities', 'History']": ['ancient egypt', 'indigenous peoples', 'social justice', 'first nations', 'middle ages'],
 "['Arts and Humanities', 'Music and Art']": ['graphic design', 'modern art', 'classical music', 'peer review', 'music production', 'contemporary art', 'music theory', 'music business'], 
 "['Arts and Humanities', 'Philosophy']": ['greek roman', 'roman mythology', 'cognitive sciences', 'time travel', 'introduction philosophy', 'introductory course'], 
 "['Business', 'Business Essentials']": ['supply chain', 'six sigma', 'project management', 'financial accounting', 'virtual students', 'excel skills', 'public speaking', 'project planning'],
 "['Business', 'Business Strategy']": ['design thinking', 'artificial intelligence', 'customer analytics', 'case studies', 'non technical', 'high level', 'project management', 'construction management', 'strategic management'],
 "['Business', 'Entrepreneurship']": ['brand management', 'renewable energy', 'case studies', 'green building','energy green', 'social impact', 'social enterprise'], 
 "['Business', 'Finance']": ['financial markets', 'private equity', 'corporate finance', 'fraud examination', 'forensic accounting'],
 "['Business', 'Leadership and Management']": ['digital transformation', 'project management', 'case studies', 'people analytics', 'food beverage', 'fashion luxury', 'product management', 'human resources', 'initiating planning'], 
 "['Business', 'Marketing']": ['social media', 'case studies', 'digital marketing', 'digital world', 'marketing analytics', 'search engine', 'search engines'], 
 "['Computer Science', 'Algorithms']": ['computer science', 'dynamic programming', 'computational thinking', 'problem solving', 'divide conquer', 'data structures', 'programming assignments', 'computer architecture', 'excel vba'],
 "['Computer Science', 'Computer Security and Networks']": ['information security'], "['Computer Science', 'Design and Product']": ['product management', 'software product', 'user experience', 'experience design', 'software development', 'design thinking', 'product manager', 'introductory course','agile practices',], 
 "['Computer Science', 'Mobile and Web Development']": ['web development', 'html css', 'front end', 'css javascript', 'java script', 'web design'], 
 "['Computer Science', 'Software Development']": ['data structures','computer science', 'programming language', 'learn python', 'functional programming', 'game development'], 
 "['Data Science', 'Data Analysis']": ['data science', 'data scientist', 'data analysis', 'big data', 'watson studio', 'final assignment', 'ibm watson', 'final project', 'open source', 'ibm cloud'], 
 "['Data Science', 'Machine Learning']": ['machine learning', 'neural networks', 'deep learning', 'neural network', 'linear algebra', 'programming assignments','artificial intelligence', 'data science'],
 "['Data Science', 'Probability and Statistics']": ['time series', 'bayesian statistics', 'real world', 'series analysis', 'probability statistics', 'basic statistics', 'data analysis', 'bayesian inference', 'practical examples'], 
 "['Health', 'Animal Health']": ['animal welfare', 'veterinary medicine', 'animal behaviour','dog emotion'],
 "['Health', 'Basic Science']": [
  'mental health',
  'stem cells',
  'chinese medicine',
  'forensic science'
],
 "['Health', 'Health Informatics']": [],
 "['Health', 'Healthcare Management']": ['drug discovery',
  'quality safety'
],
 "['Health', 'Nutrition']": ['food health',
  'eating habits',
  'healthy eating',
  'common sense',
  'excelente curso',
  'child nutrition',
  'processed foods'
],
 "['Health', 'Patient Care']": ['vital signs',
  'human body',
  'spectrum disorder',
  'autism spectrum',
  'medical terminology'
],
 "['Health', 'Psychology']": ['positive psychology',
  'first aid',
  'psychological first',
  'social psychology'
],
 "['Health', 'Public Health']": ['public health',
  'global health',
  'human rights',
  'systems thinking',
  'field epidemiology',
  'introduction epidemiology',
  'women health',
  'epidemiology public'
],
 "['Health', 'Research']": ['meta analysis',
  'clinical research',
  'clinical trials',
  'systematic review',
  'data management',
  'drug development',
  'review meta',
  'clinical trial',
  'well organized'
],
 "['Information Technology', 'Cloud Computing']": [
  'google cloud',
  'cloud platform',
  'cyber security',
  'git github',
  'cloud computing',
  'high level',
  'very good',
  'audio quality'
  ],
 "['Information Technology', 'Data Management']": [],
 "['Information Technology', 'Networking']": ['computer networking',
  'computer networks',
  'basics networking'
],
 "['Information Technology', 'Security']": ['cyber security',
  'final project'
],
 "['Information Technology', 'Support and Operations']": [
  'technical support',
  'customer service',
  'operating systems',
  'final project',
  'active directory',
  'system administration'
],
 "['Language Learning', 'Learning English']": [
  'cover letter',
  'grammar punctuation',
  'career development',
  'professional emails'
],
 "['Language Learning', 'Other Languages']": [
  'korean language',
  'seung hae',
  'hae kang',
  'este curso',
  'daily life',
  'learn korean',
  'xiaoyu liu'
],
 "['Math and Logic', 'Math and Logic']": ['data science',
  'high school',
  'math skills',
  'bayes theorem',
  'multivariate calculus',
  'linear algebra',
  'basic math'
],
 "['Personal Development', 'Personal Development']": [
  'este curso',
  'brain works',
  'personal branding',
  'high school',
  'life changing'],
 "['Physical Science and Engineering', 'Chemistry']": ['molecular spectroscopy'],
 "['Physical Science and Engineering', 'Electrical Engineering']": ['systems engineering',
  'embedded systems'
],
 "['Physical Science and Engineering', 'Environmental Science and Sustainability']": ['oil gas',
  'gas industry',
  'environmental management',
  'solar energy',
  'climate change',
  'sustainable cities',
  'operations markets',
  'renewable energy'
],
 "['Physical Science and Engineering', 'Mechanical Engineering']": ['wind energy',
  'engineering mechanics',
  'wind turbine',
  'mechanical engineering',
  'renewable energy',
  'knowledge wind'
],
 "['Physical Science and Engineering', 'Physics and Astronomy']": ['theory relativity',
  'special theory',
  'special relativity',
  'high school',
  'general theory'
],
 "['Physical Science and Engineering', 'Research Methods']": ['research methods',
  'literature review',
  'quantitative methods',
  'social sciences',
  'peer review',
  'social science',
  'research methodology',
  'research question'
],
 "['Social Sciences', 'Economics']": ['financial crisis',
  'global financial',
  'real life',
  'real world',
  'international energy',
  'decision making'
],
 "['Social Sciences', 'Education']": [
  'foundational principles'
],
 "['Social Sciences', 'Governance and Society']": ['sustainable development',
  'international relations',
  'public policy',
  'pol ticas',
  'social norms',
  'smart cities',
  'mental health',
  'este curso',
  'global diplomacy',
  'development goals'
],
 "['Social Sciences', 'Law']": ['american law',
  'human rights',
  'international criminal',
  'common law',
  'legal system',
  'derechos humanos',
  'criminal law',
  'law school',
  'children rights'
]}
 
