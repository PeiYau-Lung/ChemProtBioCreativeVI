import numpy as np
from sklearn.metrics import f1_score

def F1_micro_34569(True_label, Pred_label):
	Group = [3,4,5,6,9]
	tp = []
	fp = []
	fn = []
	predSize = []
	goldSize = []
	for i in Group:
		predSize.append(np.sum(Pred_label==i))
		goldSize.append(np.sum(True_label==i))
		tp.append(len(np.where((True_label==i)&(Pred_label==i))[0]))
		fp.append(len(np.where((True_label!=i)&(Pred_label==i))[0]))
		fn.append(len(np.where((True_label==i)&(Pred_label!=i))[0]))
	if (np.sum(tp) + np.sum(fp)) == 0.0:
		Precision = 0.0
	else:
		Precision = float(np.sum(tp)) / (np.sum(tp) + np.sum(fp))
	if (np.sum(tp) + np.sum(fn)) ==0.0:
		Recall = 0.0
	else:
		Recall = float(np.sum(tp)) / (np.sum(tp) + np.sum(fn))
	
	if Precision == 0.0 and Recall == 0.0:
		f1 = 0
	else:
		f1 = 2.0 * Precision * Recall / (Precision + Recall)
	return(f1)

def TPFP(True_label, Pred_label):
	Group = [3,4,5,6,9]
	tp = []
	fp = []
	fn = []
	for i in Group:
		tp.append( sum((True_label==i)&(Pred_label==i)) )
		fp.append( sum((True_label!=i)&(Pred_label==i)) )
		fn.append( sum((True_label==i)&(Pred_label!=i)) )

	result = [sum(tp),sum(fn),sum(fp)]
	return(result)

def TPidx(True_label, Pred_label):
	Group = [3,4,5,6,9]
	tp = []
	for i in Group:
		try:
			tp = np.hstack([tp,np.where((True_label==i)&(Pred_label==i))[0]])
		except:
			tp = np.where((True_label==i)&(Pred_label==i))[0]
	return(tp)






