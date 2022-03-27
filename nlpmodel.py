import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline

import requests
import os
import json

import pickle

data = pd.read_csv("trainingfull.csv", encoding = "ISO-8859-1")

data_clean = data.loc[:, ['Tweet', 'Sentiment']]
data_clean.head()

stopwordlist = []
with open('english') as f:
     for line in f:
        word=line.rstrip()
        stopwordlist.append(word)



vectorizer = CountVectorizer(
    analyzer = 'word',
    
    lowercase = True,
    ngram_range=(1, 2),
    stop_words = stopwordlist)


X=data_clean["Tweet"].values
y=data_clean["Sentiment"].values

pipeline_svm = make_pipeline(vectorizer, SVC(probability=True, class_weight="balanced"))

pipeline_svm.fit(X, y)

with open('classifierpickle.pkl','wb') as file:
    pickle.dump(pipeline_svm,file)
