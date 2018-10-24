import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc

#change into location of dataset, specify file name
fname = 'autoimmune_transpose.txt'

#use np.genfromtxt to read in training data, and target feature data
autoimmune_data = np.genfromtxt(fname, delimiter='\t', encoding=None, usecols=np.arange(0,9))
autoimmune_target = np.genfromtxt(fname, delimiter='\t', dtype=None, encoding=None, usecols=9)

#create models with kNN, checking the the 10-fold cross validation scores for various values of k
k_scores=[]

for i in range(1,20):
    kNN = neighbors.KNeighborsClassifier(i)
    kNN_scores = cross_val_score(kNN, autoimmune_data, autoimmune_target, cv=10)
    k_scores.append(kNN_scores.mean())

k_max = max(k_scores)
print("The highest 10-fold cross validation scores were found for k = ", k_scores.index(k_max)+1, ", which had a mean score of %.4f." % k_max, sep='')

#create models with SVM, checking the the 10-fold cross validation scores when gamma is set to 'scale' or 'auto'
s_scores=[]
s_options=['scale','auto']

for i in s_options:    
    svc = svm.SVC(gamma=i)
    svc_scores = cross_val_score(svc, autoimmune_data, autoimmune_target, cv=10)
    s_scores.append(svc_scores.mean())

s_max = max(s_scores)
print("The highest 10-fold cross validation scores were found when gamma was set to '", s_options[s_scores.index(s_max)], "', which had a mean score of %.4f." % s_max, sep='')

if k_max >= s_max:
    print("A higher mean cross validation score was obtained by the kNN algorithm.")
else:
    print("A higher mean cross validation score was obtained by the SVM algorithm.")
