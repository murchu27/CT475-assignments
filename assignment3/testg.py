import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import neighbors, svm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc

def main():
	#change into location of dataset, specify file name
	fname = 'autoimmune_transpose.txt'

	#use np.genfromtxt to read in training data, and target feature data
	autoimmune_data = np.genfromtxt(fname, delimiter='\t', encoding=None, usecols=np.arange(0,9))
	autoimmune_target = np.genfromtxt(fname, delimiter='\t', dtype=None, encoding=None, usecols=9)

	svc = svm.SVC(gamma='auto')

	#use a StratifiedKFold to generate training and testing sets
	kf = StratifiedKFold(n_splits=3)

	#loop over each fold
	for train, test in kf.split(autoimmune_data, autoimmune_target):
	    #train kNN with the given training set, and predict the class probability scores of the samples in the test set
	    sprobas_ = svc.fit(autoimmune_data[train], autoimmune_target[train]).predict(autoimmune_data[test])

if __name__ == "__main__":
	main()