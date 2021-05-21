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
from utils.equilibrage import Equilibrage

stop = list(stopwords.words('french'))

fname = "Data/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = Loader.load_pres(fname)
alltxts = np.array(alltxts)
alllab = np.array(alllabs)

params = {
    "lowercase":[False,True],
    "punct":[False,True],
    # "marker":[False,True],
    "number":[False,True],
    "stemming":[Preprocessing.stem_fr], #,Preprocessing.stem],
    # "ligne": [None,-2,0],
    "strip_accents":[False,True], # 
    "stopwords": [stop], # set(STOPWORDS)],
    "Vectorizer": [TfidfVectorizer],
    "binary": [False],
    "class_weight": [[0.1,1]],# ["balanced"],
    "max_features": [None],
    "ngram_range" : [(1,2)], # (1,1),
    "max_df" : [1.,0.08,0.005], # 0.02
    "min_df" : [1,2,10],# 5
    "clf" : [nb.MultinomialNB] # , nb.MultinomialNB,svm.LinearSVC,LogisticRegression] #nb.MultinomialNB,svm.LinearSVC,LogisticRegression]
}

# SVM => Penser à utiliser des SVM linéaire !!!!
# clf = svm.LinearSVC
# Naive Bayes
# clf = nb.MultinomialNB() # frequentiels
# regression logistique
# clf = lin.LogisticRegression()

alltxts_e, alllabs_e = Equilibrage.equilibrate_court(alltxts.reshape(-1,1), alllabs, f1 = 0.35, f2 = 0.4)
alltxts_e = list(alltxts_e.reshape(-1))
train,test = sc.gridSearch(alltxts_e,alllabs_e,params,stock = True,equilibrage_test = True)

print("Meilleurs résultats en test")
print(params.keys())
maxi = np.argmax(list(test.values()))
print(list(test.keys())[maxi])