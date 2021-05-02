# -*- coding: utf-8 -*-

import utils.scoring as sc
import sklearn.naive_bayes as nb
from sklearn import svm
from utils.utils import Loader
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
from utils.preprocessing import Preprocessing
from sklearn.linear_model import LogisticRegression

stop = list(stopwords.words('english'))
stop = list(set(stop) - {
         "no",
         "not",
         "nor"
         'ain',
         'aren',
         "aren't",
         'couldn',
         "couldn't",
         'didn',
         "didn't",
         'doesn',
         "doesn't",
         'hadn',
         "hadn't",
         'hasn',
         "hasn't",
         'haven',
         "haven't",
         'isn',
         "isn't",
         'ma',
         'mightn',
         "mightn't",
         'mustn',
         "mustn't",
         'needn',
         "needn't",
         'shan',
         "shan't",
         'shouldn',
         "shouldn't",
         'wasn',
         "wasn't",
         'weren',
         "weren't",
         'won',
         "won't",
         'wouldn',
         "wouldn't",
         'don',
         "don't",
         'should',
         "should've"})

fname = "Data/AFDmovies/movies1000/"
alltxts,alllabs = Loader.load_movies(fname)
alltxts = np.array(alltxts)
alllabs = np.array(alllabs)

params = {
    # lowercase":[False,True],
    "punct":[False,True],
    # "marker":[False,True],
    # "number":[False,True],
    "stemming":[False,Preprocessing.stem], #,Preprocessing.stem],
    "ligne": [None,-2,0],
    # "strip_accents":[False,True], # 
    "stopwords": [None,stop], # set(STOPWORDS)],
    "Vectorizer": [CountVectorizer,TfidfVectorizer],
    # "binary": [False,True],
    # "class_weight": [[0.1,1]],# ["balanced"],
    # "max_features": [None,40000, 30000],
    "ngram_range" : [(1,1),(1,2)], # (1,1),
    "max_df" : [1.,0.08,0.005], # 0.02
    "min_df" : [1,2,5],# 5
    "clf" : [svm.LinearSVC] # , nb.MultinomialNB,svm.LinearSVC,LogisticRegression] #nb.MultinomialNB,svm.LinearSVC,LogisticRegression]
}

train,test = sc.gridSearch(alltxts,alllabs,params,stock = True,test_size = 0.1,stratified=False)

print("Meilleurs résultats en test")
print(params.keys())
maxi = np.argmax(list(test.values()))
print(list(test.keys())[maxi])