# -*- coding: utf-8 -*-

from utils.utils import Loader
from sklearn.model_selection import train_test_split
from utils.preprocessing import Preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin
import numpy as np
from utils.equilibrage import Equilibrage
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold, KFold
from utils.scoring import get_vectorizer, get_classifieur
from wordcloud import WordCloud, STOPWORDS

fname = "Data/AFDmovies/movies1000/"
alltxts,alllabs = Loader.load_movies(fname)
alltxts = np.array(alltxts)
alllabs = np.array(alllabs)


stop = list(stopwords.words('english'))
stop = list(set(stop) - {
         "no",
         "not",
         "nor"
         'ain',
         'aren',
         "aren't",
         'couldn',
         "couldn't",
         'didn',
         "didn't",
         'doesn',
         "doesn't",
         'hadn',
         "hadn't",
         'hasn',
         "hasn't",
         'haven',
         "haven't",
         'isn',
         "isn't",
         'ma',
         'mightn',
         "mightn't",
         'mustn',
         "mustn't",
         'needn',
         "needn't",
         'shan',
         "shan't",
         'shouldn',
         "shouldn't",
         'wasn',
         "wasn't",
         'weren',
         "weren't",
         'won',
         "won't",
         'wouldn',
         "wouldn't",
         'don',
         "don't",
         'should',
         "should've"})

# parametres de la meilleur solution
params = {
    "punct":True,
    # "number":False,
    "stemming":False, 
    "ligne": None,
    "stopwords": None, # set(STOPWORDS)],
    "Vectorizer": TfidfVectorizer,
    "max_features": None,
    "ngram_range" : (1,2), # (1,1),
    "max_df" : 1., # 0.02
    "min_df" : 5,# 5
    "clf" : svm.LinearSVC
}

# optimisations plus tards
params["number"] = True
params["min_df"] = 7
params["binary"] = True
params["strip_accents"] = True
params["stopwords"] = {"he","it","s","of"}


#%%

vectorizer = get_vectorizer(params)

X = vectorizer.fit_transform(alltxts)

#%%
# cross validation sans équilibrage en entrainement


scoring = {'accuracy' : accuracy_score, # make_scorer(accuracy_score), 
           'precision' : precision_score, #make_scorer(precision_score),
           'recall' : recall_score, #make_scorer(recall_score), 
           'f1_score' : f1_score} #make_scorer(f1_score)}

res_cross = dict()
folds = KFold(n_splits = 20, shuffle = True)
for train, test in folds.split(X, alllabs):
    
    X_train = X[train]
    Y_train = alllabs[train]
    X_test = X[test]
    Y_test = alllabs[test]
    
    clf = get_classifieur(params,X,alllabs)
    clf.fit(X_train,Y_train)
    
    res = clf.predict(X_test)
    print(len(np.where(res==0)[0]), " on ", res.shape[0]," except :",len(np.where(Y_test==0)[0]))
    for score, element in scoring.items():
        # print(len(Y_test), len(res))
        if score != "accuracy":
            try:
                res_cross[score] += [element(Y_test,res, zero_division = 0)]
            except KeyError:
                res_cross[score] = [element(Y_test,res, zero_division = 0)]
        else:
            try:
                res_cross[score] += [element(Y_test,res)]
            except KeyError:
                res_cross[score] = [element(Y_test,res)]
    
"""
res_cross = cross_validate( clf, X, alllabs, cv=5,scoring=scoring)
"""
for score,scores in res_cross.items():
    print(score,":",np.array(scores).mean())
    
#%%
argweights = np.argsort(clf.coef_[0])
pos = dict()
neg = dict()
print("positifs")
for x in range(200):
    pos[vectorizer.get_feature_names()[argweights[-x - 1]]] = clf.coef_[0][argweights[-x - 1]]
    # print(vectorizer.get_feature_names()[argweights[-x - 1]],clf.coef_[0][argweights[-x - 1]])
print("---------")
print("negatifs")
for x in range(200):
    neg[vectorizer.get_feature_names()[argweights[x]]] = -clf.coef_[0][argweights[x]]
    # print(vectorizer.get_feature_names()[argweights[x]],clf.coef_[0][argweights[x]])

#%%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud_base = WordCloud(width = 800, height = 800, 
                background_color ='white',max_words=200, stopwords=[], normalize_plurals=False)
test = wordcloud_base.generate_from_frequencies(neg)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(test) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show() 