# import required packages
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import neighbors, svm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc



# specify file name, change into location of dataset if necessary 
fname = 'autoimmune_transpose.txt'

# use np.genfromtxt to read in training data, and target feature data
autoimmune_data = np.genfromtxt(fname, delimiter='\t', encoding=None, usecols=np.arange(0,9))
autoimmune_target = np.genfromtxt(fname, delimiter='\t', dtype=None, encoding=None, usecols=9)



# create models with kNN, checking the the 10-fold cross validation scores for various values of k
k_scores=[]

for k in range(1,20):
    kNN = neighbors.KNeighborsClassifier(k)
    kNN_scores = cross_val_score(kNN, autoimmune_data, autoimmune_target, cv=10)
	# store the mean value of these cross validation scores for comparison with other values of k
    k_scores.append(kNN_scores.mean())

# identify the highest cross validation score (say k_max), and the value of k that yielded it (say k_choose)
k_max = max(k_scores)
k_choose = k_scores.index(k_max)+1
print("The highest 10-fold cross validation scores were found for k = ", k_choose, ", which had a mean score of %.4f." % k_max, sep='')



# create models with SVM, checking the the 10-fold cross validation scores when gamma is set to 'scale' or 'auto'
s_scores=[]

for i in ['scale','auto']:
    svc = svm.SVC(gamma=i)
    svc_scores = cross_val_score(svc, autoimmune_data, autoimmune_target, cv=10)
	# store the mean value of these cross validation scores for comparison with other option for gamma 
    s_scores.append(svc_scores.mean())

# identify the highest cross validation score (say s_max), and the option that yielded it (say s_choose)
s_max = max(s_scores)
s_choose = s_options[s_scores.index(s_max)]
print("The highest 10-fold cross validation scores were found when gamma was set to '", s_choose, "', which had a mean score of %.4f." % s_max, sep='')



# determine which algorithm yielded a higher cross validation score (on average) 
if k_max >= s_max:
    print("A higher mean cross validation score was obtained by the kNN algorithm.")
else:
    print("A higher mean cross validation score was obtained by the SVM algorithm.")



# for the purpose of constructing ROC curves, construct models with each algorithm using the parameters believed to be optimal
kNN = neighbors.KNeighborsClassifier(n_neighbors=k_choose)
svc = svm.SVC(gamma=s_choose,probability=True) # note we must specify that we will want to find classifier probabilities for each sample


# in order to partition data into training/testing sets, use a StratifiedKFold with 3 splits [so that test set is 1/3 the size of the dataset]
kf = StratifiedKFold(n_splits=3)

# since our graph will contain a ROC curve for each split of the data, keep track of the current split index
i = 0
# split our dataset; train,test will both hold arrays with the indices of samples to be trained, and tested, respectively 
for train, test in kf.split(autoimmune_data, autoimmune_target):

	# train each model with the training samples, then predict the class probabilities of the testing samples
    kprobas_ = kNN.fit(autoimmune_data[train], autoimmune_target[train]).predict_proba(autoimmune_data[test])
    sprobas_ = svc.fit(autoimmune_data[train], autoimmune_target[train]).predict_proba(autoimmune_data[test])
 
    # now we compare the obtained class probabilities for the testing samples with their actual classes
	# roc_curve function returns an array of thresholds, as well as arrays with the false/true positive rates of classification using those thresholds
    kfpr, ktpr, kthresholds = roc_curve(y_true=autoimmune_target[test], y_score=kprobas_[:, 1], pos_label="positive")
    sfpr, stpr, sthresholds = roc_curve(y_true=autoimmune_target[test], y_score=sprobas_[:, 1], pos_label="positive")

	# then compute area under ROC for a single digit measure of the performance of the kNN classifier
    roc_auc = auc(kfpr, ktpr)
	# plot ROC curve for this data split, also report AUROC for this curve
    plt.figure(1)
    plt.plot(kfpr, ktpr, lw=3, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

	#repeat for SVM classifier
    roc_auc = auc(sfpr, stpr)
    plt.figure(2)
    plt.plot(sfpr, stpr, lw=3, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

	#increment current split index
    i += 1

#apply labels, legend, and grid to each graph
plt.figure(1)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for kNN Algorithm')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.4)

plt.figure(2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM algorithm')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.4)

plt.show()
