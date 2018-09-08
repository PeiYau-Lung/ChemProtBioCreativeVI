'''
This program uses predicted probability of whether CPR in L2 as meta-features, and 
makes final predictions. 
'''
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import os, sys, getopt
from PairwiseDist import *
from EdgeDistance import *
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from Scorer import F1_micro_34569
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import eli5 as el
from sklearn.feature_selection import RFECV
Show = ['method', 'transition_features', 'targets', 'feature_importances', 'decision_tree']

myscorer = make_scorer(F1_micro_34569, greater_is_better=True)


argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(argv,"hr:s:",["train=", "test="])
except getopt.GetoptError:
	print('usage: L3_Model.py -r <train> -s <test>')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('usage: L2_Model.py -r <train> -s <test>')
		sys.exit()
	elif opt in ("-r", "--train"):
		trn_prefix = arg
	elif opt in ("-s", "--test"):
		tst_prefix = arg



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

TF_trn = np.loadtxt(trn_prefix+'_labels_pair.txt')
TF_tst = np.loadtxt(tst_prefix+'_labels_pair.txt')

Features_trn = np.hstack([np.loadtxt('L2_probability_pair_withL1_'+trn_prefix+'.txt', delimiter=','),
                          np.loadtxt('L2_probability_triplet_withL1_'+trn_prefix+'.txt', delimiter=',')])
Features_trn = np.hstack([Features_trn,
                          np.loadtxt('L2_probability_pair_part2_'+trn_prefix+'.txt',delimiter=',')])
Features_trn = np.hstack([Features_trn,
                          np.loadtxt('L2_probability_triplet_part2_'+trn_prefix+'.txt',delimiter=',')])
Features_tst = np.hstack([np.loadtxt('L2_probability_pair_withL1_'+tst_prefix+'.txt', delimiter=','),
                          np.loadtxt('L2_probability_triplet_withL1_'+tst_prefix+'.txt', delimiter=',')])
Features_tst = np.hstack([Features_tst,
                          np.loadtxt('L2_probability_pair_part2_'+tst_prefix+'.txt',delimiter=',')])
Features_tst = np.hstack([Features_tst,
                          np.loadtxt('L2_probability_triplet_part2_'+tst_prefix+'.txt',delimiter=',')])

Holdout_name = []
for i in range(Features_trn.shape[0]):
	if i < Features_trn.shape[0]/2 :
		Holdout_name.append('L2pair_'+str(i+1))
	else:
		Holdout_name.append('L2trip_'+str(i+1))
 
tmp_trn = np.loadtxt('L1_probability_pair_'+trn_prefix+'.txt',delimiter=',')
tmp_tst = np.loadtxt('L1_probability_pair_'+tst_prefix+'.txt',delimiter=',')
for i in range(tmp_trn.shape[1]):
	Holdout_name.append('L1pair_'+str(i+1))
Features_trn = np.hstack([Features_trn,tmp_trn])
Features_tst = np.hstack([Features_tst,tmp_tst])
'''
Features_trn = np.loadtxt('L2_probability_pair_withL1_'+trn_prefix+'10.txt', delimiter=',')
Features_tst = np.loadtxt('L2_probability_pair_withL1_'+tst_prefix+'10.txt', delimiter=',')
'''
'''
Features_trn = np.hstack([Features_trn,np.loadtxt(trn_prefix+'_Features_pair.txt' ,delimiter=',')])
Features_tst = np.hstack([Features_tst,np.loadtxt(tst_prefix+'_Features_pair.txt' ,delimiter=',')])
'''
n_cases_trn = len(TF_trn)
n_cases_tst = len(TF_tst)
nfold = 10
os.system('echo "Starting cross validation...  '+str(Features_trn.shape[0])+' samples.."')
myscorer = make_scorer(F1_micro_34569, greater_is_better = True)
REPEAT = 1
clf_torun = ['xgb','lr','rf','et']

cvsearch = 0
for Repeat in range(REPEAT):
	kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=0*10 + 2)
	fold = 1
	CLF = {'xgb': {'cv_param': {'learning_rate': [.01, .05, .1, .5],
                                    'max_depth': [3,5,7],
                                    'min_child_weight': [1,2,3]},
                       'estimator': xgb.XGBClassifier(n_estimators=100, nthread=-1, subsample=.9, objective='multi:softmax', seed=Repeat*10+2)
                       },
               'lr': {'cv_param': {'C': [.01, .05, .1, .5, 1.0, 5.0, 10.0],
                                   'penalty': ['l1','l2']},
                      'estimator': LogisticRegression(random_state=Repeat*10+2)
                      },
               'et': {'cv_param': {'criterion': ['gini','entropy'],
                                    'max_depth': [3,5,7,None],
                                    'n_estimators': [10,20,30,50,100]}, 
                       'estimator': ExtraTreesClassifier( n_jobs=-1, random_state=Repeat*10+2)
                      }, 
               'rf': {'cv_param': {'criterion': ['gini','entropy'],
                                   'max_depth': [3,5,7,None],
                                   'n_estimators': [10,20,30,50,100]},
                      'estimator':RandomForestClassifier(n_jobs=-1, random_state=Repeat*10+2)
                      }
               }
	n_clf = len(CLF)
	Fscore_trn = np.zeros(n_clf)
	Fscore_tst = np.zeros(n_clf)
	prob_trn = np.zeros([n_clf, n_cases_trn, 10])
	pred_Y_trn = np.zeros([n_cases_trn, n_clf])
	prob_tst = np.zeros([n_clf, n_cases_tst, 10])
	
	if cvsearch ==0:	
		RANDOM_SEARCH = {}
		os.system('echo "starting random search"')
		for i in clf_torun:
			clf = CLF[i]['estimator']
			clf_param = CLF[i]['cv_param']
			RANDOM_SEARCH[i] = RandomizedSearchCV(clf, clf_param, n_iter=10, random_state=Repeat*10+2, cv=3, scoring=myscorer).fit(Features_trn,TF_trn)
		os.system('echo "random search finished"')
	
	for train_idx,valid_idx in kf.split(np.array(pair_ID_trn),pair_label_trn):	#always split on pair_label in order not to information leaking
		os.system('echo "train: '+str(len(train_idx))+'\tvalid: '+str(len(valid_idx))+'"')
		train_X = Features_trn[train_idx,:]
		train_Y = TF_trn[train_idx]
		valid_X = Features_trn[valid_idx,:]
		valid_Y = TF_trn[valid_idx]  
		for i, clf_name in enumerate(clf_torun):
			if cvsearch == 1:
				clf = CLF[clf_name]['estimator']
				clf_param = CLF[clf_name]['cv_param']
				random_search = RandomizedSearchCV(clf, clf_param, n_iter=5, random_state=Repeat*10+2, cv=2, scoring=myscorer)
				random_search.fit(train_X,train_Y)
				clf_aftersearch = random_search.best_estimator_
			else:
				clf_aftersearch = RANDOM_SEARCH[clf_name].best_estimator_
			
			selector = RFECV(clf_aftersearch,step=.2,cv=2,scoring = myscorer, verbose=0)
			selector = selector.fit(train_X,train_Y)
			selected = selector.get_support(indices=True)
			os.system('echo "selected '+str(len(selected))+' features"')
			
			if clf_name == 'xgb':
				clf_aftersearch.n_estimators = 700
			clf_aftersearch.fit(selector.transform(train_X),train_Y)
			pred_prob = clf_aftersearch.predict_proba(selector.transform(valid_X))
			prob_trn[i, valid_idx,: ] = pred_prob
			pred_Y_trn[valid_idx, i] = clf_aftersearch.predict(selector.transform(valid_X))
			prob_tst[i,:,:] = prob_tst[i,:,:] + clf_aftersearch.predict_proba(selector.transform(Features_tst))
				

		fold += 1
	for i in range(n_clf): 
		Fscore_trn[i] = F1_micro_34569(pair_label_trn,pred_Y_trn[:,i])
		Fscore_tst[i] = F1_micro_34569(pair_label_tst,prob_tst[i,:,:].argmax(1)+1)
	prob_tst = np.average(prob_tst/10, axis=0)
	#pred_Y_tst = prob_tst[0,:,:].argmax(1)+1
	pred_Y_tst = prob_tst.argmax(1)+1
	prob_trn = np.average(prob_trn,0)
	pred_Y_trn = prob_trn.argmax(1)+1
	
	#os.system('echo "using '+'--'.join(argv)+'"')
	os.system('echo "Overall '+trn_prefix+' F1: ' + str(F1_micro_34569(pair_label_trn,pred_Y_trn))+'"')	
	os.system('echo "Overall '+tst_prefix+' F1: ' + str(F1_micro_34569(pair_label_tst,pred_Y_tst))+'"')
	print(Fscore_trn,Fscore_tst)

	# generate submission
	REL = open(tst_prefix+'_ALLrelations.txt','r')
	Rel = REL.readlines()
	SUB = open(tst_prefix+'_submit1.tsv','w')
	for line_count, line in enumerate(pred_Y_tst):
		if line in [3,4,5,6,9]:
			arr = Rel[line_count].split('\t')
			SUB.write(arr[0]+'\tCPR:'+str(line)+'\t'+arr[-2]+'\t'+arr[-1])
	SUB.close()
'''
Features_name=[]
FN = open(trn_prefix+'_FeatureNames_pair.txt')
for i in FN: Features_name.append(i.split('\n')[0])
Features_name = Holdout_name + Features_name

SEN = open(tst_prefix+'_pair_sentences.txt')
OUTFP = open('FPCPR_'+tst_prefix[-3:]+'.txt','w')
OUTFN = open('FNCPR_'+tst_prefix[-3:]+'.txt','w')
selected_tmp = np.array(Features_name)[selected]
for i, line in enumerate(SEN):
	if pair_label_tst[i] in [3,4,5,6,9] and pair_label_tst[i] != pred_Y_tst[i]:
		OUTFN.write('\n**************************************************************\n')
		OUTFN.write('True:'+str(int(pair_label_tst[i]))+'\tPred:'+str(int(pred_Y_tst[i]))+'\t'+line+'\n')
		a = el.format_as_text(el.explain_prediction(clf_aftersearch,selector.transform(Features_tst)[i,], feature_names = selected_tmp,top=10,top_targets=1), 
                                      show = Show, show_feature_values=True)
		for line2 in a:
			OUTFN.write(line2)
		OUTFN.write('\n**************************************************************\n')
	if pred_Y_tst[i] in [3,4,5,6,9] and pair_label_tst[i] != pred_Y_tst[i]:
		OUTFP.write('\n**************************************************************\n')
		OUTFP.write('True:'+str(int(pair_label_tst[i]))+'\tPred:'+str(int(pred_Y_tst[i]))+'\t'+line+'\n')
		a = el.format_as_text(el.explain_prediction(clf_aftersearch,selector.transform(Features_tst)[i,], feature_names = selected_tmp,top=10,top_targets=1), 
                                      show = Show, show_feature_values=True)
		for line2 in a:
			OUTFP.write(line2)
		OUTFP.write('\n**************************************************************\n')
		
OUTFP.close()
OUTFN.close()




'''
