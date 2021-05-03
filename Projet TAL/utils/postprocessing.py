# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:35:17 2021

@author: Luc
"""
import string
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import spacy
from nltk.stem.snowball import FrenchStemmer


class Postprocessing:

    def post_major(res, fen_size = 5):
        tmp = res.copy()
        for i in range(fen_size,len(res)-fen_size):
            x, y = np.unique( tmp[i-fen_size: i+fen_size + 1], return_counts=True)
            tmp[i] = x[np.argmax(y)]

        return tmp
    
    def post_logic(res):
        res = res.copy()
        def pb_sec(res):
            cpt = [[],[]]; tmp = []; tmp2 = []
            for i in range(len(res)-1):
                if res[i] != res[i+1]:
                    tmp += [res[i]]; tmp2 += [i]
                    
                elif len(tmp) > 1:
                    cpt[0] += [tmp + [res[i]]]
                    cpt[1] += [tmp2 + [i]]
                    
                    tmp = []; tmp2 = []
                else:
                    tmp = []; tmp2 = []
            return cpt[0], cpt[1]
        
        p1, p2 = pb_sec(res)
        for i in range(len(p1)):
            if len(p2[i])%2 == 1:
                res[p2[i]] = len(p1[i])*[p1[i][0]]
            else:
                res[p2[i][:len(p2[i])//2]] = (len(p1[i])//2)*[p1[i][0]]
                res[p2[i][len(p2[i])//2:]] = (len(p1[i])//2)*[p1[i][-1]]
        return res