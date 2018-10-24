import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import neighbors, svm
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
k_choose = k_scores.index(k_max)+1
print("The highest 10-fold cross validation scores were found for k = ", k_choose, ", which had a mean score of %.4f." % k_max, sep='')

#create models with SVM, checking the the 10-fold cross validation scores when gamma is set to 'scale' or 'auto'
s_scores=[]
s_options=['scale','auto']

for i in s_options:    
    svc = svm.SVC(gamma=i)
    svc_scores = cross_val_score(svc, autoimmune_data, autoimmune_target, cv=10)
    s_scores.append(svc_scores.mean())

s_max = max(s_scores)
s_choose = s_options[s_scores.index(s_max)]
print("The highest 10-fold cross validation scores were found when gamma was set to '", s_choose, "', which had a mean score of %.4f." % s_max, sep='')

if k_max >= s_max:
    print("A higher mean cross validation score was obtained by the kNN algorithm.")
else:
    print("A higher mean cross validation score was obtained by the SVM algorithm.")

kNN = neighbors.KNeighborsClassifier(n_neighbors=k_choose)
svc = svm.SVC(gamma=s_choose,probability=True)

kf = StratifiedKFold(n_splits=3)
#train, test = kf.split(autoimmune_data, autoimmune_target)

#ktprs, kaucs = [], []
#stprs, saucs = [], []
#mean_fpr = np.linspace(0, 1, 100)

i = 0
#kprobs = []
#sprobs = []
for train, test in kf.split(autoimmune_data, autoimmune_target):
    #print("TRAIN:\n", train)
    #print("TEST:\n", test)

    kprobas_ = kNN.fit(autoimmune_data[train], autoimmune_target[train]).predict_proba(autoimmune_data[test])
    sprobas_ = svc.fit(autoimmune_data[train], autoimmune_target[train]).predict_proba(autoimmune_data[test])
    #kprobs.append(kprobas_)
    #sprobs.append(sprobas_)
 
    # Compute ROC curve and area the curve
    kfpr, ktpr, kthresholds = roc_curve(y_true=autoimmune_target[test], y_score=kprobas_[:, 1], pos_label="positive")
    sfpr, stpr, sthresholds = roc_curve(y_true=autoimmune_target[test], y_score=sprobas_[:, 1], pos_label="positive")

    #tprs.append(interp(mean_fpr, fpr, tpr))
    #ktprs.append(ktpr)
    #stprs.append(stpr)

    #ktprs[-1][0] = 0.0
    #stprs[-1][0] = 0.0

    roc_auc = auc(kfpr, ktpr)
    kaucs.append(roc_auc)
    plt.figure(1)
    plt.plot(kfpr, ktpr, lw=3, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    roc_auc = auc(sfpr, stpr)
    saucs.append(roc_auc)
    plt.figure(2)
    plt.plot(sfpr, stpr, lw=3, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))


    i += 1
    
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#         label='Chance', alpha=.8)

"""
print("[i]. Split 1                 Split 2                Split 3", end='')
for i in range(len(kprobas_)):
    print("\n%i. " % i, end='')
    for p in kprobs:
        print(p[i], end='    ')

print("\n\nSplit 3 probabilities        Actuals")
for i in range(len(kprobas_)):
    print(kprobas_[i], autoimmune_target[test[i]], sep='          ')
"""

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
