#!/bin/env python
import numpy as np
from sklearn.multiclass import OneVsOneClassifier 
from sklearn.ensemble import RandomForestClassifier
from msmbuilder.io import load_meta, load_trajs
import sys

fold = int(sys.argv[1]) 
depth = int(sys.argv[2])

crossvalid = np.load('crossvalid.npy').item()[fold]
train_index = crossvalid[0]
test_index  = crossvalid[1]

### In order to reduce memory use
train_data  = np.load('alpha_carbon/'  + str(train_index[0]) + '.npy')
train_label = np.load('macro-mapping/' + str(train_index[0]) + '.npy')
test_data   = np.load('alpha_carbon/'  + str(test_index[0]) + '.npy')
test_label  = np.load('macro-mapping/' + str(test_index[0]) + '.npy')

for i in range(1,len(train_index)):
	temp = np.load('alpha_carbon/'  + str(train_index[i]) + '.npy')
	train_data = np.append(train_data, temp, axis=0)
	#print(proc.memory_info().rss)
	del temp
	train_label  = np.append(train_label,
		np.load('macro-mapping/' + str(train_index[i]) + '.npy'), axis=0)


for i in range(1,len(test_index)):
	temp = np.load('alpha_carbon/'  + str(test_index[i]) + '.npy')
	test_data = np.append(test_data,
		np.load('alpha_carbon/'  + str(test_index[i]) + '.npy'), axis=0)
	del temp
	test_label  = np.append(test_label,
		np.load('macro-mapping/' + str(test_index[i]) + '.npy'), axis=0)


clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=0))
clf.fit(train_data,train_label)
print ('Fold: %d Depth %d Train Accu: %.3f Test Accu: %.3f' %(
	fold, depth, 
	np.sum(clf.predict(train_data) == train_label) / len(train_label),
	np.sum(clf.predict(test_data) == test_label) / len(test_label)))

del train_data,test_data,train_label,test_label

## save model
from sklearn.externals import joblib
joblib.dump(clf, 'ovo-randomforest/'+str(fold)+"_"+str(depth)+".pkl") 
