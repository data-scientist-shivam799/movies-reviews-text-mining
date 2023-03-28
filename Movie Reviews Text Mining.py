import numpy as np
import pandas as pd
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import movie_reviews

print(os.listdir(nltk.data.find("corpora")))
print(movie_reviews.categories())
print(len(movie_reviews.fileids('pos')))
print(' ')
print(movie_reviews.fileids('pos'))

neg_rev=movie_reviews.fileids('neg')
pos_rev=movie_reviews.fileids('pos')

rev=nltk.corpus.movie_reviews.words('pos/cv000_29590.txt')
print(rev)

rev_list=[]
print(len(rev_list))

for rev in neg_rev:
    rev_text_neg=rev=nltk.corpus.movie_reviews.words(rev)
    review_one_string=" ".join(rev_text_neg)
    review_one_string=review_one_string.replace(' ,',',')
    review_one_string=review_one_string.replace(' .','.')
    review_one_string=review_one_string.replace(" \'","'")
    review_one_string=review_one_string.replace("\' ","'")
    rev_list.append(review_one_string)

print(len(rev_list))

for rev in pos_rev:
    rev_text_neg = rev = nltk.corpus.movie_reviews.words(rev)
    review_one_string = " ".join(rev_text_neg)
    review_one_string = review_one_string.replace(' ,', ',')
    review_one_string = review_one_string.replace(' .', '.')
    review_one_string = review_one_string.replace(" \'", "'")
    review_one_string = review_one_string.replace("\' ", "'")
    rev_list.append(review_one_string)

print(len(rev_list))

neg_targets=np.zeros((1000,),dtype=np.int)
pos_targets=np.zeros((1000,),dtype=np.int)

target_list=[]
for neg_tar in neg_targets:
    target_list.append(neg_tar)
for pos_tar in pos_targets:
    target_list.append(pos_tar)

print(len(target_list))

y=pd.Series(target_list)
print(type(y)) # y must be pandas dataframe

print(y.head())

from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer(lowercase=True,stop_words='english',min_df=2)
X_count_vect=count_vect.fit_transform(rev_list)
print(X_count_vect.shape)

X_names=count_vect.get_feature_names()
print(X_names)

X_count_vect=pd.DataFrame(X_count_vect.toarray(),columns=X_names)
print('Shape of X_count_vect:',X_count_vect.shape)
print(X_count_vect.head())

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

X_train_cv,X_test_cv,y_train_cv,y_test_cv=train_test_split(X_count_vect,y,test_size=0.25,random_state=5)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
y_pred_gnb=gnb.fit(X_train_cv,y_train_cv).predict(X_test_cv)

from sklearn.naive_bayes import MultinomialNB
clf_cv=MultinomialNB()
clf_cv.fit(X_train_cv,y_train_cv)

y_pred_cv=clf_cv.predict(X_test_cv)
print(type(y_pred_cv))

print(metrics.accuracy_score(y_test_cv,y_pred_cv))

score_clf_cv=confusion_matrix(y_test_cv,y_pred_cv)
print(score_clf_cv)