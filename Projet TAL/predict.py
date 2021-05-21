#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils.utils import Loader
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.naive_bayes as nb
import numpy as np
from utils.equilibrage import Equilibrage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from utils.scoring import get_vectorizer, get_classifieur
from utils.postprocessing import Postprocessing
import matplotlib.pyplot as plt
from sklearn import svm
fname = "Data/AFDpresidentutf8/corpus.tache1.learn.utf8"
tname = "Data/AFDpresidentutf8/corpus.tache1.test.utf8"
fname_2 = "Data/AFDmovies/movies1000/"
tname_2 = "Data/AFDmovies/moviesTest/testSentiment.txt"


stop = list(stopwords.words('french'))

# parametres de la meilleur solution
params = {
    "lowercase":False,
    "punct":True,
    "number":False,
    "stemming":False, #,Preprocessing.stem],
    # "ligne": [None,-2,0],
    "strip_accents":True, # 
    "stopwords": stop, # set(STOPWORDS)],
    "Vectorizer": TfidfVectorizer,
    "binary": True,
    "class_weight": [1, 1],
    "max_features": None,
    "ngram_range" : (1,2), # (1,1),
    "max_df" : 0.08, # 0.02
    "min_df" : 3,# 5
    "clf" : nb.MultinomialNB
}

params_sentiments = {
    "lowercase":True,
    "punct":True,
    # "number":False,
    "stemming":False, 
    "ligne": None,
    "stopwords": None, # set(STOPWORDS)],
    "Vectorizer": TfidfVectorizer,
    "max_features": None,
    "ngram_range" : (1,2), # (1,1),
    "max_df" : 1., # 0.02
    "min_df" : 5,# 5
    "clf" : svm.LinearSVC
}

# optimisations plus tards
params["number"] = True
params["min_df"] = 7
params["binary"] = True
params["strip_accents"] = True
params["stopwords"] = {"he","it","s","of"}


def predict(X_train,Y_train, X_test, Y_test = None,  params = params, save = None, classes = [-1,1], post_processing = True, equilibrage = True):
    X_train, Y_train, X_test = np.array(X_train), np.array(Y_train), np.array(X_test)
    if Y_test is not None:
        Y_test = np.array(Y_test)
    vectorizer = get_vectorizer(params)
    if equilibrage:
        X_train, Y_train = Equilibrage.equilibrate_court(X_train.reshape(-1,1), Y_train, f1 = 0.4, f2 = 0.45)
        X_train = X_train.reshape(-1)
        
    V_train = vectorizer.fit_transform(X_train)
    V_test = vectorizer.transform(X_test)

    
    clf = get_classifieur(params,V_train,Y_train)
        
    clf.fit(V_train,Y_train)
    res = clf.predict(V_test)
    if post_processing:
        fen_size = 20 # 6
        pas = 1
        res = Postprocessing.post_major(res, fen_size = fen_size,pas = pas)
        
        fen_size = 3 # 6
        pas = 1
        res = Postprocessing.post_major(res, fen_size = fen_size,pas = pas)

    if Y_test is not None:
        scoring = {'accuracy' : accuracy_score,
                   'precision' : precision_score,
                   'recall' : recall_score,
                   'f1_score' : f1_score}
        for score, element in scoring.items():
            print(score,":",element(Y_test,res))
            
    if save is not None:
        f = open(save,"w")
        for x in res:
            if x == -1 or x == 0:
                f.write(str(classes[0]))   
            else:
                f.write(str(classes[1]))  
            f.write("\n")
    return res
        
"""      
X_train,Y_train = Loader.load_pres(fname)
X_test, _ = Loader.load_pres(tname)

result = predict(X_train, Y_train, X_test, save = "auteurs.txt", classes = ["M","C"], post_processing=True)

fig,ax = plt.subplots(figsize=(35,100)) 
ax.imshow(result.reshape(54,-1),interpolation="nearest")
"""
# plt.tight_layout()  

X_train,Y_train = Loader.load_movies(fname_2)
X_test = Loader.load_movies_test(tname_2)


result_sent = predict(X_train, Y_train, X_test, params = params_sentiments, save = "sentiments.txt", classes = ["-1","1"], post_processing=False, equilibrage = False)

