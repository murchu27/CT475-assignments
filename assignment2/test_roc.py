import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

#change into location of dataset, specify file name
fname = 'autoimmune_transpose.txt'

#use np.genfromtxt to read in training data, and target feature data
autoimmune_data = np.genfromtxt(fname, delimiter='\t', encoding=None, usecols=np.arange(0,9))
autoimmune_target = np.genfromtxt(fname, delimiter='\t', dtype=None, encoding=None, usecols=9)

cv = StratifiedKFold(n_splits=4)
classifier = svm.SVC(kernel='linear', probability=True)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(autoimmune_data, autoimmune_target):
    probas_ = classifier.fit(autoimmune_data[train], autoimmune_target[train]).predict_proba(autoimmune_data[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_true=autoimmune_target[test], y_score=probas_[:, 1], pos_label="positive")
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
