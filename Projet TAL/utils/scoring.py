# -*- coding: utf-8 -*-

from sklearn.model_selection import KFold
import itertools
import collections
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from utils.preprocessing import Preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import sklearn.naive_bayes as nb
from utils.equilibrage import Equilibrage

def kFold_scores(X,alllabs,clf,nb_splits = 2):
    scores = []
    kf = KFold(n_splits=nb_splits,shuffle=True)
    for train, test in kf.split(X):
        # print("%s %s" % (train, test))
        X_train = X[train]
        y_train = alllabs[train]
        X_test  = X[test]
        y_test  = alllabs[test]
    
        clf.fit(X_train, y_train)
        # evaluation
        scores += [clf.score(X_test,y_test)]
    return scores

def get_vectorizer(current_params):
    f = lambda x: Preprocessing.preprocessing(x,current_params)
    Vectorizer = current_params.get("Vectorizer",CountVectorizer)
    vectorizer = Vectorizer(preprocessor = f,lowercase=False,
                                token_pattern = Preprocessing.token_pattern, 
                                binary = current_params.get("binary",False), 
                                max_df = current_params.get("max_df",1),
                                min_df = current_params.get("min_df",1),
                                ngram_range = current_params.get("ngram_range",(1,1)),
                                max_features = current_params.get("max_features",None))
    return vectorizer

def get_classifieur(current_params,datax,datay):
    clf_class = current_params.get("clf",svm.LinearSVC)
    if clf_class == nb.MultinomialNB:
        class_prior = current_params.get("class_weight",None)
        if class_prior == "balanced":   
            class_prior = datax.shape[0] / (2 * np.bincount(np.where(datay == 1,1,0)))
        return clf_class(class_prior = class_prior, fit_prior = True)
    else:
        return clf_class(class_weight = current_params.get("class_weight",None), max_iter = 1000)
    

def gridSearch(datax,datay,params,stock = False,equilibrage_test = False,equilibrage_train = False):
    '''
    Parameters
    ----------
    datax
        Liste des données.
    datay 
        Liste des labels des données.
    clf_class
        Classifieur à utiliser.
    params
        Dictionnaire des parametres.
    equilibre 
        Si vrai équilibre les données en test
    Returns
    -------
    res_train 
        Dictionnaire des F1-score en train en fonction des différents parametres.
    res_test : TYPE
        Dictionnaire des F1-score en train en fonction des différents parametres..

    '''
    datay = np.array(datay)
    el = params.keys()
 
    res_test = dict()
    res_train = dict()
    size = len(list(itertools.product(*params.values())))
    for i,v in enumerate(list(itertools.product(*params.values()))):
        print(i+1,"on",size)
        tag = tuple(x if isinstance(x, collections.Hashable) else (str(x) if len(x) == 2 else "YES") for x in v)
        print(tag)
        current_params = dict(zip(el,v))
        
        # choix du classifieur
        clf = get_classifieur(current_params,datax,datay)
       
        # choix du vectorizer
        vectorizer = get_vectorizer(current_params)
        
        X = vectorizer.fit_transform(datax)
        X_train, X_test, y_train, y_test = train_test_split( X, datay, test_size=0.2, stratify = datay) 

        if equilibrage_test:
            X_test, y_test = Equilibrage.equilibrate_court(X_test,y_test,f1=1,f2=None)
            
        print("size train",X_train.shape[0],len(y_train))
        print("size test",X_test.shape[0],len(y_test))
        print("nb -1 in train : ",len(np.where(y_train == -1)[0]))
        print("nb 1 in train : ",len(np.where(y_train == 1)[0]))
        print("nb -1 in test : ",len(np.where(y_test == -1)[0]))
        print("nb 1 in test : ",len(np.where(y_test == 1)[0]))
        clf.fit(X_train, y_train)
        
        
        # Application 
        yhat_train = clf.predict(X_train)
        yhat_test = clf.predict(X_test)
       
        print("nb -1 in train predicted : ",len(np.where(yhat_train == -1)[0]))
        print("nb 1 in train predicted : ",len(np.where(yhat_train == 1)[0]))
        print("nb -1 in test predicted : ",len(np.where(yhat_test == -1)[0]))
        print("nb 1 in test predicted : ",len(np.where(yhat_test == 1)[0]))
        
        res_test[tag] = f1_score(y_test,yhat_test)
        res_train[tag] = f1_score(y_train,yhat_train)
        print("train",res_train[tag])
        print("test",res_test[tag])
    if stock:
        pickle.dump(res_train,open("train","wb"))
        pickle.dump(res_test,open("test","wb"))
    return res_train,res_test
