# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:24:22 2017

@author: Ashish
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_json('slicedReview2.json', lines=True)

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
rlist=[]
#data cleaning/stemming...
for i in range(0,5000):
    review=re.sub('^a-zA-Z',' ',dataset['text'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    rlist.append(review)
    print(i)

#adding binary value data column 'feedback'
dataset = dataset.assign(feedback=dataset['stars'].values)
dataset.feedback = dataset.feedback.replace({1:0,2:0,3:1,4:1,5:1})
    
#creating word count matrix
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=100)
X=cv.fit_transform(rlist).toarray()
y=dataset['feedback'].head(n=5000)

#sort the list
df1=pd.DataFrame(list(cv.vocabulary_.items()), columns=['Word', 'Count'])
df1=df1.sort_values('Count',ascending=False)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)