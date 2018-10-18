'''
This program uses predicted probability of whether CPR in L1 as meta-features, and 
build model. 
'''
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import os, sys, getopt
sys.path.append("./src/")
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
import pandas as pd
from Scorer import F1_micro_34569
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import time
import xgboost as xgb

myscorer = make_scorer(F1_micro_34569, greater_is_better=True)

argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(argv,"hr:s:t:l:a:",["train=", "test=", "triplet=", "IncludeL1=", "selectfromall="])
except getopt.GetoptError:
	print('usage: L2_Model.py -r <train> -s <test> -t <isUsingTriplets> -l <isIncludeL1> -a <selectfromall>')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('usage: L2_Model.py -r <train> -s <test> -t <isUsingTriplets> -l <isIncludeL1> -a <selectfromall>')
		sys.exit()
	elif opt in ("-r", "--train"):
		trainfile = arg
	elif opt in ("-t", "--triplet"):
		isTrip = int(arg)
	elif opt in ("-l", "--IncludeL1"):
		isIncludeL1 = int(arg)
	elif opt in ("-s", "--test"):
		testfile = arg
	elif opt in ("-a","--selectfromall"):
		select_all = int(arg)

os.system('echo "this is train: '+trainfile+' test: '+testfile+' triplet: '+str(isTrip)+' Include L1: '+str(isIncludeL1)+' SelectFromall: '+'"')

class ExtraTreesClassifierWithCoef(ExtraTreesClassifier):
	def fit(self, *args, **kwargs):
		super(ExtraTreesClassifierWithCoef, self).fit(*args, **kwargs)
		self.coef_ = self.feature_importances_

class RandomForestClassifierWithCoef(RandomForestClassifier):
	def fit(self, *args, **kwargs):
		super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
		self.coef_ = self.feature_importances_
tst_prefix = testfile.split('_')[0]
trn_prefix = trainfile.split('_')[0]

pair_ID_trn = []
IN = open(trn_prefix+'_pairID.txt')
for i in IN:
	pair_ID_trn.append(i.split('\n')[0])
pair_label_trn = np.loadtxt(trn_prefix+'_labels_pair.txt')

pair_ID_tst = []
IN = open(tst_prefix+'_pairID.txt')
for i in IN:
	pair_ID_tst.append(i.split('\n')[0])
pair_label_tst = np.loadtxt(tst_prefix+'_labels_pair.txt')

if isTrip:
	TF_trn = np.loadtxt(trn_prefix+'_labels_triplet.txt')
	Features_trn = np.loadtxt(trn_prefix+'_Features_triplet.txt',delimiter=',')
	TF_tst = np.loadtxt(tst_prefix+'_labels_triplet.txt')
	Features_tst = np.loadtxt(tst_prefix+'_Features_triplet.txt',delimiter=',')
	if isIncludeL1:
		Features_trn = np.hstack([Features_trn,np.loadtxt('L1_probability_triplet_'+trn_prefix+'.txt',delimiter=',')])
		Features_tst = np.hstack([Features_tst,np.loadtxt('L1_probability_triplet_'+tst_prefix+'.txt',delimiter=',')])
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
	Features_trn = np.loadtxt(trn_prefix+'_Features_pair.txt',delimiter=',')
	Features_tst = np.loadtxt(tst_prefix+'_Features_pair.txt',delimiter=',')
	if isIncludeL1:
		Features_trn = np.hstack([Features_trn,np.loadtxt('L1_probability_pair_'+trn_prefix+'.txt',delimiter=',')])
		Features_tst = np.hstack([Features_tst,np.loadtxt('L1_probability_pair_'+tst_prefix+'.txt',delimiter=',')])
print(Features_trn.shape,Features_tst.shape)
TF_bin_trn = np.in1d(TF_trn,[3,4,5,6,9])*1
TF_bin_tst = np.in1d(TF_tst,[3,4,5,6,9])*1

n_cases_trn = len(TF_trn)
n_cases_tst = len(TF_tst)
nfold = 3
print('Starting cross validation...  '+str(Features_trn.shape[0])+' training samples; '+str(Features_tst.shape[0])+' testing samples')
Start_time = time.time()
myscorer = make_scorer(F1_micro_34569, greater_is_better = True)
REPEAT = 1
if isTrip:
	clf_torun = ['rf', 'lda','et','xgb']
else:
	clf_torun = ['rf', 'lda','lr','mnb','xgb']


for Repeat in range(REPEAT):
	kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=Repeat*10 + 2)
	fold = 1
	CLF = {'et': ExtraTreesClassifierWithCoef(n_estimators = 10, n_jobs=16, random_state = Repeat*10+2),
               'lda': LinearDiscriminantAnalysis(),
               'rf': RandomForestClassifierWithCoef(n_estimators = 10, n_jobs=16, random_state = Repeat*10+2),
               'mnb': MultinomialNB(),
               'lr': LogisticRegression(random_state=Repeat*10+2,multi_class = 'multinomial',solver='saga'),
               'knn': KNeighborsClassifier(n_neighbors =15,n_jobs = 16),
               'xgb': xgb.XGBClassifier(n_estimators=700,learning_rate=0.05,subsample= .9,
                         nthread = 16,max_depth= 5,min_child_weight=1,seed = Repeat*10+2, objective='multi:softmax')}
	n_clf = len(clf_torun)
	prob_trn =  np.zeros([n_clf, n_cases_trn, 10])
	pred_Y_trn = np.zeros([n_cases_trn, n_clf])
	Holdout_prob_trn = np.zeros([len(pair_ID_trn),n_clf*10])
	Holdout_class_trn = np.zeros([len(pair_ID_trn),n_clf])
	prob_tst = np.zeros([n_clf, n_cases_tst, 10])
	pred_Y_tst = np.zeros([n_cases_tst, n_clf])		
	Holdout_prob_tst = np.zeros([len(pair_ID_tst),n_clf*10])
	Holdout_class_tst = np.zeros([len(pair_ID_tst),n_clf])


	for train_pair_idx,valid_pair_idx in kf.split(np.array(pair_ID_trn),pair_label_trn):	#always split on pair_label in order not to information leaking
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
		train_Y = TF_trn[train_idx]
		valid_X = Features_trn[valid_idx,:]
		valid_Y = TF_trn[valid_idx]
		for i, clf_name in enumerate(clf_torun):
			clf = CLF[clf_name]
			clf.fit(train_X,train_Y)
			pred_prob = clf.predict_proba(valid_X)
			prob_trn[i, valid_idx,: ] = pred_prob
			pred_Y_trn[valid_idx, i] = clf.predict(valid_X)
			pred_prob = clf.predict_proba(Features_tst)
			prob_tst[i, : , : ] = prob_tst[i, : , : ] + pred_prob
			pred_Y_tst[:, i] = clf.predict(Features_tst)

		fold += 1

	prob_tst = prob_tst / 10
	if isTrip:
		Holdout_class_trn = np.zeros([len(pair_ID_trn),n_clf])
		for k_idx, k in enumerate(pair_ID_trn):
			idx = [p for p, q in enumerate(trip_ID_trn) if q.split('_')[0] == k]
			if len(idx) == 0:
				print(k)
				sys.exit()
			for m in range(n_clf):
				class_tmp = pred_Y_trn[idx, m]
				prob_tmp = prob_trn[m, idx, :]
				Holdout_class_trn[k_idx, m] = class_tmp[prob_tmp.max(1).argmax()]	#choose the class with highest predicted probability as prediction
				Holdout_prob_trn[k_idx,(m*10):((m+1)*10)] = prob_tmp[prob_tmp.max(1).argmax(),:]
		Holdout_class_tst = np.zeros([len(pair_ID_tst),n_clf])
		for k_idx, k in enumerate(pair_ID_tst):
			idx = [p for p, q in enumerate(trip_ID_tst) if q.split('_')[0] == k]
			if len(idx) == 0:
				print(k)
				sys.exit()
			for m in range(n_clf):
				class_tmp = pred_Y_tst[idx, m]
				prob_tmp = prob_tst[m, idx, :]
				Holdout_class_tst[k_idx, m] = class_tmp[prob_tmp.max(1).argmax()]       #choose the class with highest predicted probability as prediction
				Holdout_prob_tst[k_idx,(m*10):((m+1)*10)] = prob_tmp[prob_tmp.max(1).argmax(),:]
				
	else:
		for m in range(n_clf):
			Holdout_class_trn[:, m] = pred_Y_trn[:,m]
			Holdout_prob_trn[:,(m*10):((m+1)*10)] = prob_trn[m, :, :]
			Holdout_class_tst[:, m] = pred_Y_tst[:,m]
			Holdout_prob_tst[:,(m*10):((m+1)*10)] = prob_tst[m, :, :]
	for i, clf_name in enumerate(clf_torun):
		os.system('echo "'+trn_prefix+' '+clf_name+' '+str(isTrip)+'-'+str(isIncludeL1)+' Overall F1: ' + str(F1_micro_34569(pair_label_trn,Holdout_class_trn[:,i]))+'"')	

average_prob_tst = np.zeros([Holdout_class_tst.shape[0],10])
average_prob_trn = np.zeros([Holdout_class_trn.shape[0],10])	
for m in range(n_clf):
	average_prob_tst = average_prob_tst + Holdout_prob_tst[:,(m*10):((m+1)*10)]
	average_prob_trn = average_prob_trn + Holdout_prob_trn[:,(m*10):((m+1)*10)]
average_class_tst = average_prob_tst.argmax(1) + 1
average_class_trn = average_prob_trn.argmax(1) + 1
Holdout_prob_trn = np.hstack([Holdout_prob_trn,average_prob_trn/len(clf_torun)])
Holdout_prob_tst = np.hstack([Holdout_prob_tst,average_prob_tst/len(clf_torun)])
Holdout_class_trn = np.hstack([Holdout_class_trn, average_class_trn.reshape(Holdout_class_trn.shape[0],1)])
Holdout_class_tst = np.hstack([Holdout_class_tst, average_class_tst.reshape(Holdout_class_tst.shape[0],1)])

os.system('echo "'+trn_prefix+' averaged '+str(isTrip)+'-'+str(isIncludeL1)+' Overall F1: ' + str(F1_micro_34569(pair_label_trn,average_class_trn))+'"')

Time_elapsed = time.time() - Start_time
print(str(Time_elapsed/3600)+' hours elapsed in 10-fold CV...')
if isTrip:
	if isIncludeL1:
		np.savetxt('L2_probability_triplet_withL1_'+trn_prefix+'.txt',np.hstack([Holdout_prob_trn,Holdout_class_trn]),delimiter=',',fmt='%f')
		np.savetxt('L2_probability_triplet_withL1_'+tst_prefix+'.txt',np.hstack([Holdout_prob_tst,Holdout_class_tst]),delimiter=',',fmt='%f')
	else:
		np.savetxt('L2_probability_triplet_withoutL1_'+trn_prefix+'.txt',np.hstack([Holdout_prob_trn,Holdout_class_trn]),delimiter=',',fmt='%f')
		np.savetxt('L2_probability_triplet_withoutL1_'+tst_prefix+'.txt',np.hstack([Holdout_prob_tst,Holdout_class_tst]),delimiter=',',fmt='%f')
else:
	if isIncludeL1:
		np.savetxt('L2_probability_pair_withL1_'+trn_prefix+'.txt',np.hstack([Holdout_prob_trn,Holdout_class_trn]),delimiter=',',fmt='%f')
		np.savetxt('L2_probability_pair_withL1_'+tst_prefix+'.txt',np.hstack([Holdout_prob_tst,Holdout_class_tst]),delimiter=',',fmt='%f')
	else:
		np.savetxt('L2_probability_pair_withoutL1_'+trn_prefix+'.txt',np.hstack([Holdout_prob_trn,Holdout_class_trn]),delimiter=',',fmt='%f')
		np.savetxt('L2_probability_pair_withoutL1_'+tst_prefix+'.txt',np.hstack([Holdout_prob_tst,Holdout_class_tst]),delimiter=',',fmt='%f')

nclf = int(Holdout_prob_trn.shape[1]/10)
Holdout_is2nd_trn = np.zeros(Holdout_prob_trn.shape)
Holdout_profdiff_trn = np.zeros([Holdout_prob_trn.shape[0],nclf])
Holdout_is2nd_tst = np.zeros(Holdout_prob_tst.shape)
Holdout_profdiff_tst = np.zeros([Holdout_prob_tst.shape[0],nclf])
for i in range(nclf):
	prob = Holdout_prob_trn[:,(i*10):(i+1)*10]
	for j in range(prob.shape[0]):
		a = prob[j,:].argsort()
		Holdout_profdiff_trn[j,i] = prob[j,a[-1]] - prob[j,a[-2]]
		Holdout_is2nd_trn[j,a[-2]] = 1
	prob = Holdout_prob_tst[:,(i*10):(i+1)*10]
	for j in range(prob.shape[0]):
		a = prob[j,:].argsort()
		Holdout_profdiff_tst[j,i] = prob[j,a[-1]] - prob[j,a[-2]]
		Holdout_is2nd_tst[j,a[-2]] = 1
if isTrip:
	np.savetxt('L2_probability_triplet_part2_'+trn_prefix+'.txt',np.hstack([Holdout_profdiff_trn,Holdout_is2nd_trn]),delimiter=',',fmt='%f')
	np.savetxt('L2_probability_triplet_part2_'+tst_prefix+'.txt',np.hstack([Holdout_profdiff_tst,Holdout_is2nd_tst]),delimiter=',',fmt='%f')
else:
	np.savetxt('L2_probability_pair_part2_'+trn_prefix+'.txt',np.hstack([Holdout_profdiff_trn,Holdout_is2nd_trn]),delimiter=',',fmt='%f')
	np.savetxt('L2_probability_pair_part2_'+tst_prefix+'.txt',np.hstack([Holdout_profdiff_tst,Holdout_is2nd_tst]),delimiter=',',fmt='%f')
