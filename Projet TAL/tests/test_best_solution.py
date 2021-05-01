# -*- coding: utf-8 -*-

from utils.utils import Loader
from sklearn.model_selection import train_test_split
from utils.preprocessing import Preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin
import numpy as np
from utils.equilibrage import Equilibrage
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold, KFold
from utils.scoring import get_vectorizer, get_classifieur

fname = "Data/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = Loader.load_pres(fname)
alltxts,alllabs = np.array(alltxts), np.array(alllabs)


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
    "class_weight": [0.7, 1],
    "max_features": None,
    "ngram_range" : (1,2), # (1,1),
    "max_df" : 0.08, # 0.02
    "min_df" : 10,# 5
    "clf" : nb.MultinomialNB
}

# optimisations plus tards
params["max_df"]  = 0.08
params["min_df"] = 3
params["class_weight"] = [1,1]

#%%

vectorizer = get_vectorizer(params)

X = vectorizer.fit_transform(alltxts)

#%%%

# court
alltxts_e, alllabs_e = Equilibrage.equilibrate_court(alltxts.reshape(-1,1), alllabs, f1 = 0.4, f2 = 0.45)
X_e = vectorizer.fit_transform(alltxts_e.reshape(-1))

# long
# X_e, alllabs_e = Equilibrage.equilibrate_long(X, alllabs, f1 = 0.35, f2 = 0.4)
#%%
# cross validation sans équilibrage en entrainement

 # {1:1,-1:1} # "balanced"

scoring = {'accuracy' : accuracy_score, # make_scorer(accuracy_score), 
           'precision' : precision_score, #make_scorer(precision_score),
           'recall' : recall_score, #make_scorer(recall_score), 
           'f1_score' : f1_score} #make_scorer(f1_score)}

res_cross = dict()
folds = KFold(n_splits = 10, shuffle = True)
for train, test in folds.split(X_e, alllabs_e):
    
    X_train = X_e[train]
    Y_train = alllabs_e[train]
    X_test = X_e[test]
    
    Y_test = alllabs_e[test]
    X_test, Y_test = Equilibrage.equilibrate_court(X_test, Y_test)
    clf = get_classifieur(params,X_e,alllabs_e)
    clf.fit(X_train,Y_train)
    
    res = clf.predict(X_test)
    print(len(np.where(res==-1)[0]), " on ", res.shape[0]," except :",len(np.where(Y_test==-1)[0]))
    for score, element in scoring.items():
        # print(len(Y_test), len(res))
        if score != "accuracy":
            try:
                res_cross[score] += [element(Y_test,res, zero_division = 0)]
            except KeyError:
                res_cross[score] = [element(Y_test,res, zero_division = 0)]
        else:
            try:
                res_cross[score] += [element(Y_test,res)]
            except KeyError:
                res_cross[score] = [element(Y_test,res)]
    
"""
res_cross = cross_validate( clf, X, alllabs, cv=5,scoring=scoring)
"""
for score,scores in res_cross.items():
    print(score,":",np.array(scores).mean())
    


