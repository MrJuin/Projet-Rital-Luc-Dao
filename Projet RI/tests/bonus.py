#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:48:36 2021

@author: dao
"""

import sys
sys.path.insert(1,"..")
import utils.collection as c
import utils.modeles as m
from utils.collection import IndexerSimple
from utils.metrique import Précision_moyenne, EvalIRModel
import numpy as np
from sklearn.model_selection import KFold
import itertools

base = "cisi" # cisi
col0 = c.Parser.parse("data/"+base+"/"+base+".txt")
col1 = c.QueryParser.parse("data/"+base+"/"+base+".qry", "data/"+base+"/"+base+".rel")
index_txt = IndexerSimple(col0)


def grid_search(colt,colq,index,model,params, metrique = Précision_moyenne, args = None, verbose = 0):
    """
        Parametres :
            colt : collection de texte 
            colq : collection de query
            index : indexation des textes
            model : model à tester
            params : parametres du model à tester
            metrique : metrique de test
            args : arguments de la metrique de test
            verbose : affichage du parametrage en cours d'execution si 1
        Resultat :
            - score de la metrique pour les différents paramètres
            - meilleurs paramètres
    """
    scores = []
    for v in list(itertools.product(*params.values())):
        v = dict(zip(params.keys(),v))
        if verbose:
            print(v)
        cur_model = model(index,**v)
        scores += [(v,EvalIRModel.eval(metrique,cur_model,colq,args)[0])]
    return scores, max(scores, key = lambda x:x[1])

def cross_validation(colt,colq,index,model,params, metrique = Précision_moyenne, args = None,n_splits = 3, verbose = 0):
    """
        Parametres :
            colt : collection de texte 
            colq : collection de query
            index : indexation des textes
            model : model à tester
            params : parametres du model à tester
            metrique : metrique de test
            args : arguments de la metrique de test
            n_splits : nombre de fold
            verbose : affichage du fold courant si 1, si 2 on affiche aussi les parametres en cours de test
        Resultat :
            - moyenne des meilleurs score en test pour chaque fold
            - moyenne des scores d'apprentissage pour chaque fold
            - meilleurs parametres pour chaque fold
    """
    kfold = KFold(n_splits = n_splits)
    iq = np.array(list(colq.keys()))
    train_score = dict()
    bests = []
    test_score = 0
    i = 0
    for iqtrain,iqtest in kfold.split(iq):
        if verbose: print("fold",i)
        i+=1
        colqtrain = dict(map(lambda x:(x,colq.get(x)),iq[iqtrain]))
        colqtest = dict(map(lambda x:(x,colq.get(x)),iq[iqtest]))
        scores, best = grid_search(colt, colqtrain, index, model, params, metrique, args, verbose = (verbose == 2))
        for x,y in scores:
            try:
                train_score[str(x)] += y
            except KeyError:
                train_score[str(x)] = y
        bests += [best]
        test_score += EvalIRModel.eval(metrique,model(index,**best[0]),colqtest)[0]
    for x in train_score.keys():
        train_score[x]/=n_splits
    return test_score/n_splits, train_score, bests

params_modelelangue = {
    "_lambda" : list(np.arange(0.1,1.1,0.1))
}

params_okapi = {
    "k1" : list(np.arange(0.8,1.8,0.2)),
    "b" : list(np.arange(0.5,1.5,0.25))
}
scores_modelelangue = cross_validation(col0, col1, index_txt, m.ModeleLangue, params_modelelangue, verbose = 2)


scores_okapi = cross_validation(col0, col1, index_txt, m.Okapi, params_okapi, verbose = 2)

