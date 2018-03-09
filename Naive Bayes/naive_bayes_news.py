#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:38:55 2018

@author: user
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories=['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

trainig_data=fetch_20newsgroups(subset='train', categories=categories, shuffle=True)

count_vectorizer=CountVectorizer()
xTrainCounts=count_vectorizer.fit_transform(trainig_data['data'])

tfidf=TfidfTransformer()
xTrainTfidf=tfidf.fit_transform(xTrainCounts)

model=MultinomialNB().fit(xTrainTfidf, trainig_data['target'])

new=['This has nothing to do with church or religion.', 'Sofware engineering is getting better and better.']

xNewCounts=count_vectorizer.transform(new)
xNewTfidf=tfidf.transform(xNewCounts)

predicted=model.predict(xNewTfidf)

for doc, category in zip(new, predicted):
    print("%r------->%s", (doc, trainig_data['target_names'][categories]))