# -*- coding: utf-8 -*-


import sys
sys.path.insert(1,"..")
import utils.collection as c
import utils.TextRepresenter as tr
import utils.metrique as metr
import numpy as np

def pretraitement_requete(q):
    ps = tr.PorterStemmer()
    return ps.getTextRepresentation(q)

if __name__ == "__main__":
    q = c.Query(1,"")
    q.pertinents = [1,2,5,10]
    q.pertinents_score = dict([(1,1),(2,3),(5,1),(10,2)])
    
    liste = list(range(1,11))

    assert metr.Précision.evalQuery(liste, q, [5]) == 3/5 
    assert metr.Rappel.evalQuery(liste, q, [5]) == 3/4
    assert metr.F_mesure.evalQuery(liste, q, [5, 0.5]) == (1+0.5**2)*(3/5*3/4)/((0.5**2)*3/5 + 3/4)
    DCG = 1 + 3/np.log2(2) + 1/np.log2(5) + 2/np.log2(10)
    IDCG = 3 + 2/np.log2(2) + 1/np.log2(3) + 1/np.log2(4)
    assert metr.NDCG.evalQuery(liste, q) == DCG/IDCG
    assert metr.reciprocal_rank.evalQuery(liste, q) == 1/1
    assert metr.Précision_moyenne.evalQuery(liste, q) == (1+1+3/5+4/10)/len(q.pertinents)
    
    # exemple vu en cours
    q.pertinents = [1,4,5,8] 
    assert (metr.Précision_interpolée.evalQuery(liste,q)[0] == np.array([0.25,0.5,0.75,1])).all()
    assert (metr.Précision_interpolée.evalQuery(liste,q)[1] == np.array([1.0,0.6,0.6,0.5])).all()