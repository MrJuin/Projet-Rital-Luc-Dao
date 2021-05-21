import numpy as np
import matplotlib.pyplot as plt

import codecs
import re
import os.path

# Chargement des données:
    
class Loader():
    def load_pres(fname):
        alltxts = []
        alllabs = []
        s=codecs.open(fname, 'r','utf-8') # pour régler le codage
        while True:
            txt = s.readline()
            if(len(txt))<5:
                break
            lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
            txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
            if lab.count('M') >0:
                alllabs.append(-1)
            else: 
                alllabs.append(1)
            alltxts.append(txt)
        return alltxts,alllabs
    
    def load_movies(path2data): # 1 classe par répertoire
        alltxts = [] # init vide
        labs = []
        cpt = 0
        for cl in os.listdir(path2data): # parcours des fichiers d'un répertoire
            for f in os.listdir(path2data+cl):
                txt = open(path2data+cl+'/'+f).read()
                alltxts.append(txt)
                labs.append(cpt)
            cpt+=1 # chg répertoire = cht classe
            
        return alltxts,labs
    
    def load_movies_test(fname):
        alltxts = []
        s=codecs.open(fname, 'r','utf-8') # pour régler le codage
        txt = True
        while txt:
            txt = s.readline()
            alltxts.append(txt)

        return alltxts