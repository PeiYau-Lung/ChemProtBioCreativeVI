'''
This program builds model to get predicted probability of whether CPR. 
The predicted probability will later be used as additional features in L2 model. 
'''
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import os, sys, getopt
sys.path.append("./src/")
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
import pandas as pd
from Scorer import F1_micro_34569
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import time


argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(argv,"hr:s:t:a:",["train=", "test=", "triplet=", "selectfromall="])
except getopt.GetoptError:
	print('usage: L1_Model.py -r <trian> -s <test> -t <isUsingTriplets> -a <selectfromall>')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('usage: L1_Model.py -r <train> -s <test> -t <isUsingTriplets> -a <selectfromall>')
		sys.exit()
	elif opt in ("-r", "--train"):
		trainfile = arg
	elif opt in ("-s", "--test"):
		testfile = arg
	elif opt in ("-t", "--triplet"):
		isTrip = int(arg)
	elif opt in ("-a","--selectfromall"):
		select_all = int(arg)		

os.system('echo "this is train: '+trainfile+' test: '+testfile+' triplet: '+str(isTrip)+' SelectFromall: '+str(select_all)+'"')

tst_prefix = testfile.split('_')[0]
trn_prefix = trainfile.split('_')[0]

class GradientBoostingClassifierWithCoef(xgb.XGBClassifier):
	def fit(self, *args, **kwargs):
		super(GradientBoostingClassifierWithCoef, self).fit(*args, **kwargs)
		self.coef_ = self.feature_importances_

class ExtraTreesClassifierWithCoef(ExtraTreesClassifier):
        def fit(self, *args, **kwargs):
                super(ExtraTreesClassifierWithCoef, self).fit(*args, **kwargs)
                self.coef_ = self.feature_importances_

class RandomForestClassifierWithCoef(RandomForestClassifier):
	def fit(self, *args, **kwargs):
		super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
		self.coef_ = self.feature_importances_

pair_ID_trn = []
IN = open(trn_prefix+'_pairID.txt')
for i in IN:
	pair_ID_trn.append(i.split('\n')[0])

pair_ID_tst = []
IN = open(tst_prefix+'_pairID.txt')
for i in IN:
	pair_ID_tst.append(i.split('\n')[0])

if isTrip:
	TF_trn = np.loadtxt(trn_prefix+'_labels_triplet.txt')
	TF_tst = np.loadtxt(tst_prefix+'_labels_triplet.txt')
	Features_trn = np.loadtxt(trn_prefix+'_Features_triplet.txt',delimiter=',')
	Features_tst = np.loadtxt(tst_prefix+'_Features_triplet.txt',delimiter=',')
	trip_ID_trn = []
	IN = open(trn_prefix+'_tripletID.txt')
	for i in IN:
		trip_ID_trn.append(i.split('\n')[0])
	trip_ID_tst = []
	IN = open(tst_prefix+'_tripletID.txt')
	for i in IN:
		trip_ID_tst.append(i.split('\n')[0])

else:
	TF_trn = np.loadtxt(trn_prefix+'_labels_pair.txt')
	TF_tst = np.loadtxt(tst_prefix+'_labels_pair.txt')
	Features_trn= np.loadtxt(trn_prefix+'_Features_pair.txt',delimiter=',')
	Features_tst= np.loadtxt(tst_prefix+'_Features_pair.txt',delimiter=',')	

TF_bin_trn = np.array([1*(i in [3,4,5,6,9]) for i in TF_trn])
is10 = np.array([1*(i == 10) for i in TF_trn])
TF_bin_tst = np.array([1*(i in [3,4,5,6,9]) for i in TF_tst])

n_cases_trn = len(TF_trn)
n_cases_tst = len(TF_tst)
nfold = 10
print('Starting cross validation...  '+str(Features_trn.shape[0])+' training samples; '+str(Features_tst.shape[0])+' testing samples')
Start_time = time.time()
myscorer = make_scorer(F1_micro_34569, greater_is_better = True)
REPEAT = 1
clf_torun = ['xgb', 'rf', 'et']

for Repeat in range(REPEAT):
	kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=Repeat*10 + 2)
	fold = 1
	CLF = {'xgb': GradientBoostingClassifierWithCoef(n_estimators=100,learning_rate=0.05,subsample= .9,nthread = 16,max_depth= 5,min_child_weight=1,seed = Repeat*10+2),
               'rf': RandomForestClassifierWithCoef(n_estimators = 10, n_jobs=16, random_state = Repeat*10+2),
               'et': ExtraTreesClassifierWithCoef(n_estimators = 10, n_jobs=16, random_state = Repeat*10+2)}
	n_clf = len(CLF)
	prob_trn =  np.zeros([n_cases_trn,n_clf])
	Holdout_pred_trn = np.zeros([n_cases_trn,n_clf])
	Holdout_prob_trn = np.zeros([n_cases_trn,n_clf])
	prob_tst =  np.zeros([nfold,n_cases_tst,n_clf])
	if select_all == 1:
		SELECTOR = {}
		for i in clf_torun:
			SELECTOR[i] = RFECV(CLF[i],step=.1,cv=3,scoring = 'f1', verbose=0).fit(Features_trn,TF_bin_trn)

	for train_pair_idx,valid_pair_idx in kf.split(np.array(pair_ID_trn),np.loadtxt(trn_prefix+'_labels_pair.txt')):#always split on pair_label in order not to information leaking
		if isTrip:
			tmp1 = np.array(pair_ID_trn)[train_pair_idx]
			train_idx = np.array([i for i, j in enumerate(trip_ID_trn) if j.split('_')[0] in tmp1])
			tmp2 = np.array(pair_ID_trn)[valid_pair_idx]
			valid_idx = np.array([i for i, j in enumerate(trip_ID_trn) if j.split('_')[0] in tmp2])
		else:
			train_idx = train_pair_idx
			valid_idx = valid_pair_idx
		os.system('echo "train: '+str(len(train_idx))+'\tvalid: '+str(len(valid_idx))+'"')
		train_X = Features_trn[train_idx,:]
		train_Y = TF_bin_trn[train_idx]
		valid_X = Features_trn[valid_idx,:]
		valid_Y = TF_bin_trn[valid_idx]
		for i, clf_name in enumerate(clf_torun):
			clf = CLF[clf_name]
			if select_all ==1:
				selector = SELECTOR[clf_name]
				selected = selector.get_support(indices=True)
			elif select_all ==0:
				selector = RFECV(clf,step=.1,cv=3,scoring = 'f1', verbose=0)
				selector = selector.fit(train_X,train_Y)
				selected = selector.get_support(indices=True)
		
			clf = CLF[clf_name]
			if clf_name == 'xgb': clf.n_estimators = 700
			clf.fit(selector.transform(train_X),train_Y)
			pred_prob = clf.predict_proba(selector.transform(valid_X))
			prob_trn[valid_idx,i] = pred_prob[:,1]
			pred_prob = clf.predict_proba(selector.transform(Features_tst))
			prob_tst[fold-1,:,i] = pred_prob[:,1]

		fold += 1
	
	Holdout_prob_tst = np.mean(prob_tst, axis = 0)
	Holdout_pred_tst = 1*(Holdout_prob_tst>0.5)
	Holdout_pred_trn = 1*(prob_trn>0.5)
	for i, clf_name in enumerate(clf_torun):
		os.system('echo "'+trn_prefix+' '+clf_name+'-'+str(isTrip)+' Overall F1_bin: ' + str(f1_score(TF_bin_trn,1*(prob_trn[:,i]>0.5)))+'"')	
		prob_trn[:,i] = rankdata(prob_trn[:,i],method='max')
		prob_trn[:,i] = prob_trn[:,i]*1.0/prob_trn[:,i].max()
		Holdout_prob_tst[:,i] = rankdata(Holdout_prob_tst[:,i],method='max')
		Holdout_prob_tst[:,i] = Holdout_prob_tst[:,i]*1.0/Holdout_prob_tst[:,i].max()
	

Time_elapsed = time.time() - Start_time
print(str(Time_elapsed/3600)+' hours elapsed in 10-fold CV...')


if isTrip:
	np.savetxt('L1_probability_triplet_'+trn_prefix+'.txt',np.hstack([prob_trn,Holdout_pred_trn]),delimiter=',',fmt='%f')
	np.savetxt('L1_probability_triplet_'+tst_prefix+'.txt',np.hstack([Holdout_prob_tst,Holdout_pred_tst]),delimiter=',',fmt='%f')
else:
	np.savetxt('L1_probability_pair_'+trn_prefix+'.txt',np.hstack([prob_trn,Holdout_pred_trn]),delimiter=',',fmt='%f')
	np.savetxt('L1_probability_pair_'+tst_prefix+'.txt',np.hstack([Holdout_prob_tst,Holdout_pred_tst]),delimiter=',',fmt='%f')




