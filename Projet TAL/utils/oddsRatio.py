# -*- coding: utf-8 -*-

import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# oddsratio
class OddsRatioCloud:
    def __init__(self,datax,datay,lower = True, # si on lower
                 stopwords = None, # mots à ne pas regarder
                 letters_numbers = False, # si supprime caractères non lettres/chiffres
                 numbers = True, # si supprime nombres
                 min_appear = 0, # nombre de fois que le mot doit apparaitre (borne non incluse)
                 singular = True, # si passage au singulier des mots 
                 lambd = 1 # on suppose lambd élément dans chaque classe
                 ):
        
        
        self.datax = datax
        self.datay = datay
        
        self.lower = lower
        self.stopwords = stopwords
        self.letters_numbers = letters_numbers
        self.numbers = numbers
        self.min_appear = min_appear

        self.singular = singular
        self.lambd = lambd
        
    def traitement(self,x):
        if self.lower: 
            x = x.lower()
        if self.letters_numbers:
            x = re.sub(r'[\W_ ]+','',x)
        if not self.numbers:
            x = re.sub(r"\d+","",x)
        if self.singular:
            if len(x) > 1 and x[-1] == 's' and x[-2] != 's':
                x = x[:-1]
        return x
    
    def _add_to_dicts(self,x,y,yplus,ymoins,dictio_yplus,dictio_ymoins,dictio_total):
        try:
            dictio_total[x] += 1
        except KeyError:
            dictio_total[x] = 1
        if y == yplus:
            try:
                dictio_yplus[x] += 1
            except KeyError:
                dictio_yplus[x] = 1
        elif ymoins is None or ymoins == y:
            try:
                dictio_ymoins[x] += 1
            except KeyError:
                dictio_ymoins[x] = 1
                
    def _get_result(self,dictio_total,dictio_yplus,dictio_ymoins,total):
        res = []
        tot_yplus = sum(dictio_yplus.values())
        tot_ymoins = sum(dictio_ymoins.values())
        for m,tot in dictio_total.items():
            nbm = dictio_ymoins.get(m,0)
            nbp = dictio_yplus.get(m,0)
            if tot <= self.min_appear:
                continue
            p = (nbp+self.lambd)/(tot_yplus+total*self.lambd)
            q = (nbm+self.lambd)/(tot_ymoins+total*self.lambd) # suppose apparait au moins une fois dans l'autre
            res += [(m,(p*(1-q))/(q*(1-p)),tot)]
        res = sorted(res, key = lambda x:x[1],reverse = True)
        return res
    
    def print_res(self,res):
        def transforme(tab):
            res = dict()
            for mot,odds,nb in tab:
                res[mot] = odds
            return res

        res = transforme(res)
        # utilise frequence des mots dans l'ordre
        wordcloud_base = WordCloud(width = 800, height = 800, 
                background_color ='white',max_words=200)

        test = wordcloud_base.generate_from_frequencies(res)
        plt.figure(figsize = (8, 8), facecolor = None) 
        plt.imshow(test) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
  
        plt.show() 
        
    def init(self,
            yplus, # le tag à regardé
            ymoins = None, # le tag à regardé si yplus faux, on met None si tout le reste
            print_res = True # affichage du resultats
            
            ):
        dictio_yplus = dict()
        dictio_ymoins = dict()
        dictio_total = dict()
        
        dictio_yplus_bi = dict()
        dictio_ymoins_bi = dict()
        dictio_total_bi = dict()
        for i,(x,y) in enumerate(zip(self.datax,self.datay)):
            
            if self.stopwords is not None and x.lower() in self.stopwords:
                continue
            x = self.traitement(x)
            if x == "":
                continue
            self._add_to_dicts(x, y, yplus, ymoins, dictio_yplus, dictio_ymoins, dictio_total)
            if i == len(self.datax) - 1: continue   
            x2 = self.datax[i+1]
            y2 = self.datay[i+1]
            if y2 != y: continue # pas dans la même phrase
            if self.stopwords is not None and x2.lower() in self.stopwords:
                continue
            x2 = self.traitement(x2)
            if x2 == "":
                continue
            bigram = x + " " + x2
            self._add_to_dicts(bigram, y, yplus, ymoins, dictio_yplus_bi, dictio_ymoins_bi, dictio_total_bi)
        
        total = len(dictio_total_bi) + len(dictio_total)
        res = self._get_result(dictio_total,dictio_yplus,dictio_ymoins,total)
        res_2 = self._get_result(dictio_total_bi,dictio_yplus_bi,dictio_ymoins_bi,total)
        if print_res:
            res_final = res +res_2
            self.print_res(res_final)
        return res, res_2