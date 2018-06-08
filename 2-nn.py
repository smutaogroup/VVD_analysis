#!/bin/env python
import numpy as np
from sklearn.neural_network import MLPClassifier
from msmbuilder.io import load_meta, load_trajs
import sys

fold = int(sys.argv[1]) 
alpha = 10 ** int(sys.argv[2])
alpha_log = int(sys.argv[2])

crossvalid = np.load('crossvalid.npy').item()[fold]
train_index = crossvalid[0]
test_index  = crossvalid[1]

### In order to reduce memory use
train_data  = np.load('alpha_carbon_fs/'  + str(train_index[0]) + '.npy')
train_label = np.load('macro-mapping/' + str(train_index[0]) + '.npy')
test_data   = np.load('alpha_carbon_fs/'  + str(test_index[0]) + '.npy')
test_label  = np.load('macro-mapping/' + str(test_index[0]) + '.npy')

for i in range(1,len(train_index)):
	temp = np.load('alpha_carbon_fs/'  + str(train_index[i]) + '.npy')
	train_data = np.append(train_data, temp, axis=0)
	#print(proc.memory_info().rss)
	del temp
	train_label  = np.append(train_label,
		np.load('macro-mapping/' + str(train_index[i]) + '.npy'), axis=0)


for i in range(1,len(test_index)):
	temp = np.load('alpha_carbon_fs/'  + str(test_index[i]) + '.npy')
	test_data = np.append(test_data,
		np.load('alpha_carbon_fs/'  + str(test_index[i]) + '.npy'), axis=0)
	del temp
	test_label  = np.append(test_label,
		np.load('macro-mapping/' + str(test_index[i]) + '.npy'), axis=0)


clf = MLPClassifier(alpha=alpha, hidden_layer_sizes=(400, 200, 100,), random_state=1)
clf.fit(train_data,train_label)
print ('Fold: %d Alpha %.5f Train Accu: %.3f Test Accu: %.3f' %(
	fold, alpha_log, 
	np.sum(clf.predict(train_data) == train_label) / len(train_label),
	np.sum(clf.predict(test_data) == test_label) / len(test_label)))

del train_data,test_data,train_label,test_label

## save model
from sklearn.externals import joblib
joblib.dump(clf, 'neuralnetwork/'+str(fold)+"_"+str(alpha_log)+".pkl") 
