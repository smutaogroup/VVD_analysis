#!/bin/env python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from msmbuilder.io import load_meta, load_trajs
import sys
from sklearn.externals import joblib

depth = 9 
meta, all_data  = load_trajs('alpha_carbon/')
meta, all_label = load_trajs('macro-mapping/')
all_data_one = np.concatenate(list(all_data.values()))
all_label_one = np.concatenate(list(all_label.values()))

clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=0))
clf.fit(all_data_one, all_label_one)
print (' Depth %d Train Accu: %.3f' %(
	depth, np.sum(clf.predict(all_data_one) == all_label_one) / len(all_label_one)))

## save model
joblib.dump(clf, 'ovo-randomforest/final_es100_'+str(depth)+".pkl") 
