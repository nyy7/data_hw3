#!/usr/bin/python

import random
import urllib2
import numpy 
import pip, sys, os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn import metrics

#cats = ['alt.atheism','sci.space']
newsgroups_total = fetch_20newsgroups()
data = newsgroups_total.data
target = newsgroups_total.target

# Classification
def classifier_compare(clf_name):
	pre_score = 0
	recall_score = 0
	f1_score = 0

	# generate K-fold iterator
	kf = KFold(len(target),n_folds=5,shuffle=False,random_state=None)
	
	#transform text to counts
	count_vect = CountVectorizer()
	X_counts = count_vect.fit_transform(data)
		
	# transform counts to frequencies
	X_tf = TfidfTransformer(use_idf=False).fit_transform(X_counts)
		 
	# feature selection
	select = SelectPercentile(chi2, percentile = 10)
	X_fs = select.fit_transform(X_tf,target)

	for train_index,test_index in kf:
		target_train = target[train_index]
		target_test = target[test_index]
		
		if clf_name == "tree":
			clf = DecisionTreeClassifier(random_state=0)
		 	X_fs2 = X_fs.toarray()
			data_train = X_fs2[train_index]
			data_test = X_fs2[test_index]
			clf_test = clf.fit(data_train,target_train)
			target_pre = clf_test.predict(data_test)
	 	elif clf_name == "KNN":
			clf = KNeighborsClassifier(n_neighbors=3)
			X_fs2 = X_fs
			data_train = X_fs2[train_index]
			data_test = X_fs2[test_index]
			clf_test = clf.fit(data_train,target_train)
			target_pre = clf_test.predict(data_test)
	  	elif clf_name == "MNB":
			clf = MultinomialNB()
			X_fs2 = X_fs
			data_train = X_fs2[train_index]
			data_test = X_fs2[test_index]
			clf_test = clf.fit(data_train,target_train)
			target_pre = clf_test.predict(data_test)
		
		# precision score accumulation in each class
		pre_element = metrics.precision_score(target_test,target_pre,average="weighted")
		#if pre_score == []:
		#	pre_score = pre_element
		#else:
		#	pre_score = [a + b for a, b in zip(pre_score, pre_element)]
		pre_score += pre_element
		# recall score accumulation in each class
		recall_element = metrics.recall_score(target_test,target_pre,average="weighted")
		#if recall_score == []:
		#	recall_score = recall_element
		#else:
		#	recall_score = [a + b for a, b in zip(recall_score,recall_element)]
		recall_score += recall_element
		# f1 score accumulation in each class
		f1_element = metrics.f1_score(target_test,target_pre,average="weighted")
		#if f1_score == []:
		#	f1_score = f1_element
		#else:
		#	f1_score = [a + b for a, b in zip(f1_score,f1_element)]
		f1_score += f1_element
	# calculate the average score for each class
	#pre_score = [a / 5 for a in pre_score]
	#recall_score = [a / 5 for a in recall_score]
	#f1_score = [a / 5 for a in f1_score]
	pre_score = pre_score/5
	recall_score = recall_score/5
	f1_score = f1_score/5
	# print results
	print "---------------------------------------"
	print "Result of " + clf_name + "classifier: "
	print "precision score = ", pre_score
	print "recall_score = ", recall_score
	print "f1_score = ", f1_score
	print "---------------------------------------"

#tree
classifier_compare("tree")
#KNN
classifier_compare("KNN")
#MNB
classifier_compare("MNB")


