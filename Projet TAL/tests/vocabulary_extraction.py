# -*- coding: utf-8 -*-
from utils.utils import Loader
from utils.preprocessing import Preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils.oddsRatio import OddsRatioCloud
from time import time
import spacy
from nltk.corpus import stopwords

fname = "Data/AFDpresidentutf8/corpus.tache1.learn.utf8"
train_x,train_y = Loader.load_pres(fname)

stop = list(stopwords.words('french')) # + ['cet', 'cette', 'là']
params = {
    "lowercase":False,
    "punct":False,
    "marker":False,
    "number":False,
    "stemming": Preprocessing.lem, # Preprocessing.stem,
    "ligne": None,
    "strip_accents":False,
    "stopwords": stop # set(stop)
}
f = lambda x: Preprocessing.preprocessing(x,params)
#%%


vectorizer = CountVectorizer(preprocessor = f,lowercase=False,token_pattern = Preprocessing.token_pattern)
t = time() ; X = vectorizer.fit_transform(train_x); print(time() - t)
print("nombres de mots différents :",len(vectorizer.get_feature_names()))
print("nombre de mots :",X.sum())

#%%
datax_tr = list(map(f, train_x))

# word cloud des frequences
def wcloud(x):
    wordcloud_base = WordCloud(width = 800  , height = 800 , background_color ='white',\
                               max_words=100,stopwords = [],collocations = False,\
                               normalize_plurals = False, include_numbers = True)
    
    wordcloud = wordcloud_base.generate(" ".join(x))
    
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) ; plt.axis("off") ; plt.tight_layout(pad = 0)
    plt.show()
    return wordcloud

wordcloud = wcloud(datax_tr)

# Distribution des mots selon les étiquettes
chi_x = np.array(datax_tr)[np.where(np.array(train_y) ==  1)[0]]
wcloud(chi_x)

mit_x = np.array(datax_tr)[np.where(np.array(train_y) == -1)[0]] 
wcloud(mit_x)

#%% 
# oddsratio cloud
def oddsratiocloud(x,y):
    datax = []; datay = []
    for phrase, l in zip(x,y):
        mots = phrase.split(" ")
        datax += mots
        datay += len(mots)*[l]
    
    min_ap = 30
    cloud = OddsRatioCloud(datax, datay, 
                         lower=False, 
                         stopwords=[], 
                         letters_numbers = False, 
                         numbers = False,
                         min_appear = min_ap,
                         lambd = 10e-3)
    
    res, res_2 = cloud.init(1)
    res, res_2 = cloud.init(-1)            
    
oddsratiocloud(datax_tr, train_y)

#%%
# distribution d'apparitions des mots
res = dict()
for phrase in datax_tr:
    mots = phrase.split(" ")
    for mot in mots:
        try:
            res[mot]+=1
        except KeyError:
            res[mot]=1
        
plt.figure(figsize=(20,10))
plt.plot(list(res.values()))
plt.show()
