# -*- coding: utf-8 -*-

from utils.utils import Loader
from sklearn.model_selection import train_test_split
from utils.preprocessing import Preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin
import numpy as np
from utils.equilibrage import Equilibrage
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold
from utils.scoring import get_vectorizer, get_classifieur
                         
#%%
fname = "Data/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = Loader.load_pres(fname)
alllabs = np.array(alllabs)


stop = list(stopwords.words('french'))

# parametres de la meilleur solution
params = {
    "lowercase":False,
    "punct":True,
    "number":True,
    "stemming":False, #,Preprocessing.stem],
    # "ligne": [None,-2,0],
    "strip_accents":True, # 
    "stopwords": stop, # set(STOPWORDS)],
    "Vectorizer": CountVectorizer,
    "binary": True,
    # "class_weight": "balanced",
    "max_features": None,
    "ngram_range" : (1,2), # (1,1),
    # "max_df" : 0.005, # 0.02
    # "min_df" : 1,# 5
    "clf" : svm.LinearSVC
}



vectorizer = get_vectorizer(params)

X = vectorizer.fit_transform(alltxts)

#%%
# cross validation sans équilibrage en entrainement

scoring = {'accuracy' : accuracy_score, # make_scorer(accuracy_score), 
           'precision' : precision_score, #make_scorer(precision_score),
           'recall' : recall_score, #make_scorer(recall_score), 
           'f1_score' : f1_score} #make_scorer(f1_score)}

res_cross = dict()
folds = StratifiedKFold(n_splits = 5)
for train, test in folds.split(X, alllabs):
    clf = get_classifieur(params,alltxts,alllabs)
    X_test, Y_test = Equilibrage.remove_prioritaire(X[test],alllabs[test],marge=0)
    clf.fit(X[train],alllabs[train])
    res = clf.predict(X_test)
    for score, element in scoring.items():
        try:
            res_cross[score] += [element(Y_test,res)]
        except KeyError:
            res_cross[score] = [element(Y_test,res)]
    
    
"""
res_cross = cross_validate( clf, X, alllabs, cv=5,scoring=scoring)
"""
for score,scores in res_cross.items():
    print(score,":",np.array(scores).mean())


#%%
# cross validation avec équilibrage en entrainement

params["class_weight"] = "balanced" # dict({1:0.1,-1:0.9})

scoring = {'accuracy' : accuracy_score, # make_scorer(accuracy_score), 
           'precision' : precision_score, #make_scorer(precision_score),
           'recall' : recall_score, #make_scorer(recall_score), 
           'f1_score' : f1_score} #make_scorer(f1_score)}

res_cross = dict()
folds = StratifiedKFold(n_splits = 5)
for train, test in folds.split(X, alllabs):
    clf = get_classifieur(params,alltxts,alllabs)
    X_train, Y_train = Equilibrage.remove_prioritaire(X[train],alllabs[train],marge=0)
    X_test, Y_test = Equilibrage.remove_prioritaire(X[test],alllabs[test],marge=0)
    print("from", X[train].shape[0], "to", X_train.shape[0])
    clf.fit(X_train,Y_train)
    
    res = clf.predict(X_test)
    for score, element in scoring.items():
        try:
            res_cross[score] += [element(Y_test,res)]
        except KeyError:
            res_cross[score] = [element(Y_test,res)]
    
    
"""
res_cross = cross_validate( clf, X, alllabs, cv=5,scoring=scoring)
"""
for score,scores in res_cross.items():
    print(score,":",np.array(scores).mean())

params["class_weight"] = None
