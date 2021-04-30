# -*- coding: utf-8 -*-

from utils.equilibrage import Equilibrage
import numpy as np
from utils.utils import Loader

fname = "Data/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = Loader.load_pres(fname)
alllabs = np.array(alllabs)
alltxts = np.array(alltxts)
#%%
datax, datay = Equilibrage.remove_prioritaire(alltxts,alllabs)

[ind1,ind2] = np.unique(datay)
print(ind1,":",sum(np.where(datay==ind1,1,0)),"<=",sum(np.where(alllabs==ind1,1,0)))
print(ind2,":",sum(np.where(datay==ind2,1,0)),"<=",sum(np.where(alllabs==ind2,1,0)))
