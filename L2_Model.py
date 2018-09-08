'''
This program uses predicted probability of whether CPR in L1 as meta-features, and 
build model. 
'''
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import os, sys, getopt
from PairwiseDist import *
from EdgeDistance import *
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
import pandas as pd
from Scorer import F1_micro_34569
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import TruncatedSVD
import time
from imblearn.under_sampling import OneSidedSelection,TomekLinks
import eli5 as el
US = 0

PD = PairwiseDist()
ED = EdgeDistance()
td_list = list(ED.HieDep.keys())
td_list.sort()
myscorer = make_scorer(F1_micro_34569, greater_is_better=True)
'''
Features:
0. length of sentence
1. steps in sp3 (sp of PROT1 to PROT2)
2. position of p1
3. position of p2
4. type of protein
5. Number of protein
6-15. types of CPR-words mentioned in [p1-3, p2+3]
'''

argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(argv,"hr:s:t:l:a:f:",["train=", "test=", "triplet=", "IncludeL1=", "selectfromall=","tfidf="])
except getopt.GetoptError:
	print('usage: L2_Model.py -r <train> -s <test> -t <isUsingTriplets> -l <isIncludeL1> -a <selectfromall> -f <tfidf>')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('usage: L2_Model.py -r <train> -s <test> -t <isUsingTriplets> -l <isIncludeL1> -a <selectfromall> -f <tfidf>')
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
	elif opt in ("-f","--tfidf"):
		istfidf = int(arg)

os.system('echo "this is train: '+trainfile+' test: '+testfile+' triplet: '+str(isTrip)+' Include L1: '+str(isIncludeL1)+' SelectFromall: '+str(select_all)+' istfidf: '+str(istfidf)+'"')

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
	if istfidf == 0:
		Features_trn = np.loadtxt(trn_prefix+'_Features_pair.txt',delimiter=',')
		Features_tst = np.loadtxt(tst_prefix+'_Features_pair.txt',delimiter=',')
	else:
		tmpin = open(trn_prefix+'_pair_sentences.txt')
		sen_trn = []
		for i in tmpin: sen_trn.append(i.split('\n')[0])
		tmpin = open(tst_prefix+'_pair_sentences.txt')
		sen_tst = []
		for i in tmpin: sen_tst.append(i.split('\n')[0])
		start_time = time.time()
		tfidf =  Pipeline([('tfidf', TfidfVectorizer(ngram_range = (1, 2), analyzer = "word", max_features = 3000)),('best', TruncatedSVD(n_components = 1000))])
		Counts = CountVectorizer(ngram_range = (1, 2), analyzer = "word", max_features = 3000)
		combined_features = FeatureUnion([("tfidf", tfidf), ("Counts", Counts), ])
		Features_trn = combined_features.fit_transform(sen_trn)
		Features_tst = combined_features.fit_transform(sen_tst)
	if isIncludeL1:
		Features_trn = np.hstack([Features_trn,np.loadtxt('L1_probability_pair_'+trn_prefix+'.txt',delimiter=',')])
		Features_tst = np.hstack([Features_tst,np.loadtxt('L1_probability_pair_'+tst_prefix+'.txt',delimiter=',')])
print(Features_trn.shape,Features_tst.shape)
TF_bin_trn = np.in1d(TF_trn,[3,4,5,6,9])*1
TF_bin_tst = np.in1d(TF_tst,[3,4,5,6,9])*1

n_cases_trn = len(TF_trn)
n_cases_tst = len(TF_tst)
nfold = 10
print('Starting cross validation...  '+str(Features_trn.shape[0])+' training samples; '+str(Features_tst.shape[0])+' testing samples')
Start_time = time.time()
myscorer = make_scorer(F1_micro_34569, greater_is_better = True)
REPEAT = 1
if istfidf == 0:
	clf_torun = ['et', 'rf', 'lda','xgb']
else:
	clf_torun = ['et', 'rf', 'lda']

for Repeat in range(REPEAT):
	kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=Repeat*10 + 2)
	fold = 1
	CLF = {'xgb': GradientBoostingClassifierWithCoef(n_estimators=100,learning_rate=0.05,subsample= .9,
                         nthread = 16,max_depth= 5,min_child_weight=1,seed = Repeat*10+2, objective='multi:softmax'),
               'et': ExtraTreesClassifierWithCoef(n_estimators = 10, n_jobs=16, random_state = Repeat*10+2),
               'lda': LinearDiscriminantAnalysis(),
               'rf': RandomForestClassifierWithCoef(n_estimators = 10, n_jobs=16, random_state = Repeat*10+2),
               'mnb': MultinomialNB()}
	n_clf = len(CLF)
	prob_trn =  np.zeros([n_clf, n_cases_trn, 10])
	pred_Y_trn = np.zeros([n_cases_trn, n_clf])
	Holdout_prob_trn = np.zeros([len(pair_ID_trn),n_clf*10])
	Holdout_class_trn = np.zeros([len(pair_ID_trn),n_clf])
	prob_tst = np.zeros([n_clf, n_cases_tst, 10])
	pred_Y_tst = np.zeros([n_cases_tst, n_clf])		
	Holdout_prob_tst = np.zeros([len(pair_ID_tst),n_clf*10])
	Holdout_class_tst = np.zeros([len(pair_ID_tst),n_clf])

	if select_all == 1 and istfidf == 0:
		SELECTOR = {}
		for i in clf_torun:
			SELECTOR[i] = RFECV(CLF[i],step=.2,cv=3,scoring = myscorer, verbose=0).fit(Features_trn,TF_trn)

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
		if US == 1 :
			UnderSample = TomekLinks(return_indices=True,random_state=Repeat*10+2, n_jobs = 16)
			train_X, tmpy, train_idx = UnderSample.fit_sample(train_X, (train_Y==10)*1)
			train_Y = train_Y[train_idx]
			#for i in range(10): print(sum(train_Y==(i+1)))  
		for i, clf_name in enumerate(clf_torun):
			clf = CLF[clf_name]
			if select_all == 1 and istfidf == 0:
				selector = SELECTOR[clf_name]
				selected = selector.get_support(indices=True)
			elif select_all ==0 and istfidf == 0:
				selector = RFECV(clf,step=.5,cv=2,scoring = myscorer, verbose=0)
				selector = selector.fit(train_X,train_Y)
				selected = selector.get_support(indices=True)
		
			clf = CLF[clf_name]
			if clf_name == 'xgb': clf.n_estimators = 700
			clf.fit(selector.transform(train_X),train_Y)
			pred_prob = clf.predict_proba(selector.transform(valid_X))
			prob_trn[i, valid_idx,: ] = pred_prob
			pred_Y_trn[valid_idx, i] = clf.predict(selector.transform(valid_X))
			pred_prob = clf.predict_proba(selector.transform(Features_tst))
			prob_tst[i, : , : ] = prob_tst[i, : , : ] + pred_prob
			pred_Y_tst[:, i] = clf.predict(selector.transform(Features_tst))

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
		os.system('echo "'+tst_prefix+' '+clf_name+' '+str(isTrip)+'-'+str(isIncludeL1)+' Overall F1: ' + str(F1_micro_34569(pair_label_tst,Holdout_class_tst[:,i]))+'"')	

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
os.system('echo "'+tst_prefix+' averaged '+str(isTrip)+'-'+str(isIncludeL1)+' Overall F1: ' + str(F1_micro_34569(pair_label_tst,average_class_tst))+'"')

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


if isTrip:
	sys.exit()
Features_name=[]
FN = open(trn_prefix+'_FeatureNames_pair.txt')
for i in FN: Features_name.append(i.split('\n')[0])
Features_name = Features_name + ['xgb_prob','rf_prob','et_prob','xgb_pred','rf_pred','et_pred']

SEN = open(tst_prefix+'_pair_sentences.txt')
OUTFP = open('FPCPR_'+tst_prefix[-3:]+'.txt','w')
OUTFN = open('FNCPR_'+tst_prefix[-3:]+'.txt','w')
selected_tmp = np.array(Features_name)[selected]
Show = ['method', 'transition_features', 'targets', 'feature_importances', 'decision_tree']
clf = CLF['xgb']
clf.n_estimators=700
for i, line in enumerate(SEN):
	if pair_label_tst[i] in [3,4,5,6,9] and pair_label_tst[i] != Holdout_class_tst[i,-1]:
		OUTFN.write('\n**************************************************************\n')
		OUTFN.write('True:'+str(int(pair_label_tst[i]))+'\tPred:'+str(int(Holdout_class_tst[i,-1]))+'\t'+line+'\n')
		a = el.format_as_text(el.explain_prediction(clf,selector.transform(Features_tst)[i,], feature_names = selected_tmp,top=10), show = Show, show_feature_values=True)
		for line2 in a:
			OUTFN.write(line2)
		OUTFN.write('\n**************************************************************\n')
	if Holdout_class_tst[i,-1] in [3,4,5,6,9] and pair_label_tst[i] != Holdout_class_tst[i,-1]:
		OUTFP.write('\n**************************************************************\n')
		OUTFP.write('True:'+str(int(pair_label_tst[i]))+'\tPred:'+str(int(Holdout_class_tst[i,-1]))+'\t'+line+'\n')
		a = el.format_as_text(el.explain_prediction(clf,selector.transform(Features_tst)[i,], feature_names = selected_tmp,top=10), show = Show, show_feature_values=True)
		for line2 in a:
			OUTFP.write(line2)
		OUTFP.write('\n**************************************************************\n')

OUTFP.close()
OUTFN.close()
