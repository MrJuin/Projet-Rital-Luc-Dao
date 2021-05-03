from utils.utils         import Loader
from utils.equilibrage   import Equilibrage
from utils.utils         import Loader
from utils.preprocessing import Preprocessing
from utils.postprocessing import Postprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection         import train_test_split
from sklearn import linear_model as lin
from sklearn import svm
import sklearn.naive_bayes as nb

from wordcloud   import WordCloud
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from time import time
import spacy
import numpy as np
import pickle
fname = "Data/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = Loader.load_pres(fname)

stop = list(stopwords.words('french')) + ['cet', 'cette', 'là']

# parametres de la meilleur solution
stop = list(stopwords.words('french')) + ['cet', 'cette', 'là']
params = {
    "lowercase":True,
    "punct":True,
    "marker":True,
    "number":True,
    "stemming": Preprocessing.stem,
    "ligne": None,
    "strip_accents":True,
    "stopwords": set(stop)
}
f = lambda x: Preprocessing.preprocessing(x,params)
t = time(); data_x = list(map(f, alltxts)); print("temps 1 :",time()-t)
vectorizer = CountVectorizer(preprocessor = None,lowercase=False,token_pattern = Preprocessing.token_pattern)
t = time(); X = vectorizer.fit_transform(data_x) ; print("temps 2 :",time() -t)

# train test split sans équilibrage
t = time()
clf = svm.LinearSVC()
X_train, X_test, y_train, y_test = train_test_split( X, alllabs, test_size=0.4, random_state=0) 
clf.fit(X_train, y_train)
print("temps 3:",time() - t)


res = clf.predict(X_test)
print(clf.score(X_test,y_test))
print(np.unique(res,return_counts=True))

#%%
res = clf.predict(X_train)
res2 = Postprocessing.post_major(res, fen_size = 3)

res3 = Postprocessing.post_logic(res)

print(np.sum(np.array(y_train) == np.ones(len(y_train)))/len(res2))

#%%
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