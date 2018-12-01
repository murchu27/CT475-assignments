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

#report which algorithm has higher cross validation score
if k_max >= s_max:
    print("A higher mean cross validation score was obtained by the kNN algorithm.")
else:
    print("A higher mean cross validation score was obtained by the SVM algorithm.")

#keep the models that generated the highest cross validation scores for each algorithm
kNN = neighbors.KNeighborsClassifier(n_neighbors=k_choose)
svc = svm.SVC(gamma=s_choose,probability=True)

#use a StratifiedKFold to generate training and testing sets
kf = StratifiedKFold(n_splits=3)

#we now use the split method of StratifiedKfold to split our dataset into folds
#with 1/3 of the dataset to be put aside for testing in each fold

num_pos=0 #counts the number of positive samples in the test set of each fold
num_neg=0 #counts the number of negative samples in the test set of each fold
fold = 0 #keeps track of the current ROC fold for labelling the graph

#create arrays to hold the tpr arrays and the auc scores for each fold 
ktprs, kaucs = [], []
stprs, saucs = [], []

#generate a linear space between 0 and 1 to use as fpr points; we will generate tpr points using interpolation
mean_fpr = np.linspace(0, 1, 20)

#loop over each fold
for train, test in kf.split(autoimmune_data, autoimmune_target):
    #count number of positive and negative samples
    for k in test:
        if autoimmune_target[k] == "positive":
            num_pos+=1
        else:
            num_neg+=1

    #train kNN with the given training set, and predict the class probability scores of the samples in the test set
    kprobas_ = kNN.fit(autoimmune_data[train], autoimmune_target[train]).predict_proba(autoimmune_data[test])
    
    # Compute ROC curve for kNN, store true positive rates for this curve so that mean tprs can be calculated later
    kfpr, ktpr, kthresholds = roc_curve(y_true=autoimmune_target[test], y_score=kprobas_[:, 1], pos_label="positive")
    ktprs.append(interp(mean_fpr, kfpr, ktpr)) #interpolate some tpr values at the values in mean_fpr so that mean tprs can be calculated later
    ktprs[-1][0] = 0.0 #manually set the tpr at (0,0) to zero, in case interpolation hasn't done this already 
    
    # Compute AUROC scores for kNN, then report this when plotting the kNN ROC curve for this fold 
    roc_auc = auc(kfpr, ktpr)
    kaucs.append(roc_auc) #store the AUROC score for this fold
    plt.figure(1) #figure 1 is kNN curves
    plt.plot(kfpr, ktpr, lw=1, alpha=0.3,
             label="ROC fold %d (AUC = %0.3f) \n#Positive test samples: %d \n#Negative test samples: %d" % (fold, roc_auc, num_pos, num_neg))

    #repeat all these steps for SVM
    sprobas_ = svc.fit(autoimmune_data[train], autoimmune_target[train]).predict_proba(autoimmune_data[test])
    sother = svc.fit(autoimmune_data[train], autoimmune_target[train]).predict(autoimmune_data[test])
    sfpr, stpr, sthresholds = roc_curve(y_true=autoimmune_target[test], y_score=sprobas_[:, 1], pos_label="positive")
    stprs.append(interp(mean_fpr, sfpr, stpr))
    stprs[-1][0] = 0.0
    roc_auc = auc(sfpr, stpr)
    saucs.append(roc_auc)
    plt.figure(2) #figure 2 is SVM curves
    plt.plot(sfpr, stpr, lw=1, alpha=0.3,
             label="ROC fold %d (AUC = %0.3f) \n#Positive test samples: %d \n#Negative test samples: %d" % (fold, roc_auc, num_pos, num_neg))

    #next ROC fold, reset num_pos & num_neg, increment fold
    num_pos,num_neg=0,0
    fold += 1

#now that we have interpolated the same number of tpr values for each fold, we can easily find the mean values at each index
#first for kNN
mean_tpr = np.mean(ktprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr) #also calculate AUROC score for these mean tpr/fpr values

#find how close to the ideal it gets
dists_to_ideal = []
for i in range(len(mean_tpr)):
	dists_to_ideal.append(np.sqrt((1-mean_tpr[i])**2 + (0-mean_fpr[i])**2))

#plot mean curve
plt.figure(1)
plt.plot(mean_fpr, mean_tpr, color='b', #finally, plot the mean curve
         label="Mean ROC (AUC = %0.3f) \nMinimum distance to ideal: %0.3f" % (mean_auc, min(dists_to_ideal)),
         lw=3, alpha=.8)


#then for SVM
mean_tpr = np.mean(stprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
dists_to_ideal = []
for i in range(len(mean_tpr)):
	dists_to_ideal.append(np.sqrt((1-mean_tpr[i])**2 + (0-mean_fpr[i])**2))
plt.figure(2)
plt.plot(mean_fpr, mean_tpr, color='b',
         label="Mean ROC (AUC = %0.3f) \nMinimum distance to ideal: %0.3f" % (mean_auc, min(dists_to_ideal)),
         lw=3, alpha=.8)


# Format the graphs appropriately, and then display them
plt.figure(1)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for kNN Algorithm")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.4)

plt.figure(2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for SVM Algorithm")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.4)

plt.show()
