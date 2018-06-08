#!/bin/env python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.externals import joblib

clf = joblib.load('ovo-randomforest/final_9.pkl')

def non_zero(singlearray,desire_imp):
	a = np.array([[i,v] for i,v in enumerate(singlearray)])
	return (a[a[:,1] > desire_imp][:,0])

def ovo(clf, desire_imp):
	total = []
	for m in clf.estimators_:
		total.append(non_zero(m.feature_importances_, desire_imp))
	total = np.concatenate(total).astype(int)
	total = np.unique(total)
	return total

select_feature = ovo(clf,0.0001)

for i in range(12):
	a = np.load('alpha_carbon/'+str(i)+'.npy')
	a = a[:,select_feature]
	np.save('alpha_carbon_fs/'+str(i)+'.npy',a)

