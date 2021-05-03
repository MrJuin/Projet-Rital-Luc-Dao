# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:43:28 2021

@author: Luc
"""
import numpy as np


def sous_graph(model, query, n, k):
    S = np.array(model.getRanking(query))[:n]
    V = set(S[:,0])
    for doc, score in S:
        Out = model.index.getHyperlinksFrom(doc)
        In  = model.index.getHyperlinksTo(doc)
        V = V.union(list(Out.keys()))
        V = V.union(np.random.choice(list(In.keys()), k))
    return V
    
def page_ranking(model, query, n, k,  d, nbiter_max = 100, epsilon = 1e-100):
    """
    entrées:
        k : nombre de documents aléatoire à prendre
        n : nombre de documents à considérer
        d : probabilité
    """
    S = sous_graph(model, query, n, k)
    Pi = np.ones(len(S))*(1/len(S))
    s = list(S)
    A = np.zeros((len(s), len(s)))
    def find_num(x):
        return s.index(str(x))
    
    for doc in S: #Création de la matrice
        nb_liens = model.index.getHyperlinksFrom(doc)
        nb_liens = np.array(list(nb_liens.items()),dtype = np.intc)
        nb_liens = nb_liens[np.isin(nb_liens[:,0], s)]
        tmp = 1/sum(nb_liens[:,1])
        indices = list(map(find_num, nb_liens[:,0]))
        A[find_num(doc)][indices] = nb_liens[:,1]*tmp
            
    for i in range(nbiter_max): #Itérations
        P_new = np.dot(Pi, A)*d + (1-d)
        P_new /= np.sum(P_new)
        P_old = Pi
        Pi  = P_new
        print(np.sum(np.abs(P_new-P_old)))
        if np.sum(np.abs(P_new-P_old)) < epsilon :
            print('eject', i)
            break
        
    sort = np.flip(np.argsort(Pi))
    return np.array(s)[sort]



def page_ranking_dict(model, query, n, k,  d, nbiter_max = 100, epsilon = 1e-100):
    S = sous_graph(model, query, n, k)
    
    s = dict(zip(S, [1/len(S)]*len(S)))
    p = dict()
    
    for doc in S:
        try :
            nb_liens = model.index.getHyperlinksFrom(doc)
            nb_liens = np.array(list(nb_liens.items()),dtype = np.intc)
            tmp = np.isin(nb_liens[:,0], list(S))
            nb_liens = nb_liens[tmp]
            p[doc] = sum(nb_liens[:,1])
        except (KeyError, IndexError):
            p[doc] = 0
    
    for i in range(nbiter_max):
        s_new = dict(zip(S, [0]*len(S)))
        
        for j in S:
            try:
                In = model.index.getHyperlinksTo(j)        
                In = np.array(list(In.items()))
                In = In[np.isin(In[:,0], list(S))]
                for doc, nbr in In:
                    s_new[j] += (int(nbr)/p[doc])*s[doc]
            except KeyError:
                pass
            
            s_new[j] = s_new[j]*d + (1-d)
                
        norm = np.sum(list(s_new.values()))
        for j in S:
            s_new[j] /= norm
        
        l1 = np.array(list(s_new.values()))
        l2 = np.array(list(s.values()))
        s = s_new
        
        print(np.sum(np.abs(l1-l2)))
        if np.sum(np.abs(l1-l2)) < epsilon :
            print('eject', i)
            break
    
    sort = np.flip(np.argsort(list(s.values())))
    return np.array(list(s.keys()))[sort]