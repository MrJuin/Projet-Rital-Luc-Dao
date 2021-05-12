#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:56:40 2021

@author: dao
"""

import utils.TextRepresenter as tr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

class EvalMesure:
    def evalQuery(liste, query, args):
        pass


class EvalIRModel:    
    def eval_query(mesure,model,query,ps,args = None):
        q = ps.getTextRepresentation(query.text)
        ranking = np.array(model.getRanking(q))
        ranking = ranking[:,0]
    
        m = mesure.evalQuery(ranking,query,args)
        return m
        
    # renvoie la moyenne de l'erreur et sa variance
    def eval(mesure, model, col_q, args = None):
        ps = tr.PorterStemmer()
        res = list(map(lambda query: EvalIRModel.eval_query(mesure,model,query,ps,args),col_q.values()))
        return np.mean(res), np.std(res,ddof = 1)
    
    # affiche le graphe de précision interpolée pour nb query de col_q
    def precision_interpolée_graph(model,col_q,nb=0):
        if nb == 0:
            nb = len(col_q)
            
        def pretraitement_requete(q):
            ps = tr.PorterStemmer()
            return ps.getTextRepresentation(q)
        keys = list(col_q.keys())
    
        if nb != len(col_q):
            np.random.shuffle(keys)
        i = 0
        while i!=nb:
            query = col_q[keys[i]]
            i+=1
            if len(query.pertinents) == 0: # pas de pertinents, courbe inutile
                print("requete",keys[i],"=> pas de pertinents disponibles")    
                continue
            
            q = pretraitement_requete(query.text)
            ranking = model.getRanking(q)
            ranking = np.array(ranking)[:,0]
            x,y = Précision_interpolée.evalQuery(ranking,query)

            plt.figure()
            plt.title("Precision Interpolée pour la requête "+query.id)
            plt.plot(x,y)
            
    def significativité(alpha, mesure, model, model_test, col_q, args):
        '''
            T-Test
            Renvoie Faux si les modeles sont différents, Vrai sinon
            alpha entre 0 et 100
        '''
        s = len(col_q)
        m, std = EvalIRModel.eval(mesure, model, col_q, args)
        m_test, std_test = EvalIRModel.eval(mesure, model_test, col_q, args)
        # print(m_test,m,std_test,std)
        t = (m_test-m)/(np.sqrt((std/np.sqrt(s))**2 + (std_test/np.sqrt(s))**2))
        # print(t)
        p = stat.t.cdf(np.abs(t),(s - 1)*2)
        cv = stat.t.ppf(1 - alpha/100, (s - 1) * 2)
        # print(p, (1 - alpha/100), cv)
        if np.abs(t) > cv: return False
        else: return True
        

class Précision(EvalMesure):
    def evalQuery(liste, query, args = [5]):
        
        '''
            Compute the Précision at rank k
            If rank 0, the rank is the number of pertinent elements
            args[0] : k
        '''
        if args[0] == 0: 
            
            args[0] = len(query.pertinents)
            if args[0]==0: return 1
        return np.sum(np.isin(liste[:args[0]],query.pertinents))/args[0]
    
    def allEvalQuery(liste,query, args = None):
        '''
            Compute for all k the Query
        '''
        return np.cumsum(np.where(np.isin(liste,query.pertinents),1,0))/\
            range(1,len(liste)+1)
    
class Rappel(EvalMesure):
    def evalQuery(liste, query, args = [5]):
        '''
            Compute the Rappel at rank k
            If rank 0, the rank is the number of pertinent elements
            args[0] : k
        '''
        if args[0] == 0: args[0] = len(query.pertinents)
        if len(query.pertinents) == 0: return 1
        return np.sum(np.isin(liste[:args[0]],query.pertinents))/\
            len(query.pertinents)
            
    def allEvalQuery(liste,query,args = None):
        '''
            Compute for all k the Rappel
        '''
        if len(query.pertinents) == 0: return [1]*len(liste)
        return np.cumsum(np.where(np.isin(liste,query.pertinents),1,0))/\
            len(query.pertinents)
    
class F_mesure(EvalMesure):
    def evalQuery(liste,query,args = [4, 0.5]):
        '''
            Compute the F_mesure at rank k
            If rank 0, the rank is the number of pertinent elements
            args[0] : k
            args[1] : beta
        '''
        if args[0] == 0: 
            args[0] = len(query.pertinents)
        p = Précision.evalQuery(liste,query,[args[0]])
        r = Rappel.evalQuery(liste,query,[args[0]])
        if ((args[1]**2)*p+r) == 0: return 0
        return (1+args[1]**2)*(p*r)/((args[1]**2)*p+r)
    
    def allEvalQuery(liste,query,args = [0.5]):
        '''
            Compute for all k the F_mesure
            args[0] : beta
        '''
        p = Précision.allEvalQuery(liste,query)
        r = Rappel.allEvalQuery(liste,query)
        return (1+args[0]**2)*(p*r)/((args[0]**2)*p+r)
    
    
class Précision_moyenne(EvalMesure):
    def evalQuery(liste,query, args = None):
        R = np.where(np.isin(liste,query.pertinents),1,0)
        P = Précision.allEvalQuery(liste, query)
        if len(query.pertinents) == 0:
            return 1
        return np.sum(R*P)/len(query.pertinents)

        
class reciprocal_rank(EvalMesure):
    def evalQuery(liste, query, args = None):
        """
            query est une requête 
            liste est une liste de ranking de document pour chaque requêtes
        """
        ens = np.where(np.isin(liste, query.pertinents))[0]
        if len(ens) == 0: return 0
        return 1/(ens[0] +1)
    
class NDCG(EvalMesure):
    def __tryvalue(x,l):
        try:
            return l[x]
        except KeyError:
            return 0
        
    def evalQuery(liste, query, args = None):
        """
            query est une requêtes avec des documents pertinents 
            liste est un ranking des documents pour la requête par un modèle
        """
        if len(query.pertinents) == 0: return 1
        r = np.array(list(
                map(lambda x: NDCG.__tryvalue(x,query.pertinents_score),liste)
                ))
        #r = np.where(np.isin(liste, query.pertinents),1,0)
        DCG =  r[0] + np.sum(r[1:] / np.log2(np.arange(2,len(r)+1)))
        i = sorted(list(query.pertinents_score.values()),reverse=True)
        IDCG = i[0] + np.sum(i[1:] / np.log2(np.arange(2,len(i)+1)))
        return DCG/IDCG
    
class Précision_interpolée(EvalMesure):
    def evalQuery(liste,query,args = None):
        """
            Renvoie 2 listes avec pts les precisions et values les rappels associés
        """
        p = np.array(Précision.allEvalQuery(liste,query))

        r = np.array(Rappel.allEvalQuery(liste,query))
        pts,ids = np.unique(r,return_index=True)
        values = []
        for i in ids:
            v = r[i]
            values+=[np.max(p[np.where(r>=v)[0]])]
        return pts,values