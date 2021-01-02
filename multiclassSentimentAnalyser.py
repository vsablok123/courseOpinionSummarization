# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import token

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score


coursereviews = pd.read_csv("Coursera_reviews.csv") 


#Number of 1 rating sents - 50000
#Number of 2 rating sents - 50000
#Number of 3 rating sents - 110000
#Number of 4 rating sents - 110000
#Number of 5 rating sents - 110000
#deliberate inbalanced dataset to better distinguish between 3,4, and 5 ratings. rather than 1 or 2.

#ratio of test examples 10000*5

