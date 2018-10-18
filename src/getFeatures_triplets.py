'''
This program extracts features for triplets.
'''

import os, sys, getopt
sys.path.append("./src/")
from PairwiseDist import *
from EdgeDistance import *
import numpy as np
from nltk.tokenize import word_tokenize

PD = PairwiseDist()
ED = EdgeDistance()
td_list = list(ED.HieDep.keys())
td_list.sort()

argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(argv,"hp:",["prefix="])
except getopt.GetoptError:
	print('usage: getFeatures_triplets.py -p <prefix>')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('usage: getFeatures_triplets.py -p <prefix>')
		sys.exit()
	elif opt in ("-p", "--prefix"):
		prefix = arg
	


SEN = open(prefix + '_tagged.txt','r')
sen_ID = []
sen_length = {}
sen_sentence = {}
for line in SEN:
	arr = line.strip().split('\t')
	sen_ID.append(arr[0])
	sen_length[arr[0]] = len(arr[-1].split(' '))
	sen_sentence[arr[0]] = [' '] + arr[-1].split(' ')

PAIR = open(prefix + '_pairs_label.txt','r')
pair_ID = []
pair_label = []
for line in PAIR:
	arr = line.strip().split('\t')
	pair_ID.append(arr[0])
	pair_label.append(int(arr[1].split(':')[1]))
try:
	pair_label[pair_label.index(0)] = 10
except:
	print('no relation has label 0')

IW_type_dic = {}
IW_dir_dic = {}
IW_type_fn = open('./src/IW_CPR_09_2017.txt')
IW_dic = {}
IW_count = 0
for line in IW_type_fn:
	if line.startswith('#') or len(line) < 5:
		continue
	else:
		arr = line.strip().split('\t')
		try:
			IW_type_dic[arr[0]] = [int(arr[2]),int(arr[3])]       # using types defined both for CPR and for PPI
		except:
			IW_type_dic[arr[0]] = [int(arr[2]),10]

		if arr[1] == 'Y':
			IW_dir_dic[arr[0]] = 2
		elif arr[1] == 'N':
			IW_dir_dic[arr[0]] = 1
		elif arr[1] == 'S':
			IW_dir_dic[arr[0]] = 0
		IW_dic[arr[0]] = IW_count
		IW_count += 1

ProteinName = []
PNF = open(prefix+'_ProteinNames.txt', 'r')
for i in PNF:
	ProteinName.append(i.strip().split('\t')[0])

SP = open(prefix+'_shortest_path.txt','r')

pair_ID_trip = []
IW = []
TF = []
sp_type = []
DIR = []
IW_type = []
IW_cluster = []
sen_len = []
HofP = [] # high:2 low:1   high means the more than 2 proteins are included from three words before the 1st element of triplets to three words after last element of triplet
NofP = [] # number of proteins are included from three words before the 1st element of triplets to three words after last element of triplet
IW_dir = []
Prot_type = []  #0: gene-N  1: gene-Y
Significant = []
p1_minus = []
p2_minus = []
isAdjacent = []

Interactor = []
D1 = []
D2 = []
Order = []	# avb:1 vab:2 abv:3
Comma = []	# nn:1 yn:2 ny:3 yy:4
Not = []	# y:2 n:1
Breaker = []	# y:2 n:1
Cdtnal = []	# y:2 n:1
Prep = []	# list of proposition
#Prep_dic = {}
Prep_list = []
Which = []	# y:2 n:1
But = []	# y:2 n:1
NofIW = []	# high:2 low:1
Prep_count = 0

TD_SP = []	# a list of three numpy array, and each one np array has dim of nCases*nTypeDependencies

trip_ID = []
trip_ID_list = []
#trip_count = 0
for line_count, line in enumerate(SP):
	arr = line.strip().split('\t')
	if line.startswith('#'):
		trip_ID = arr[1]
		pair_ID_trip.append(arr[1].split('_')[0])
		trip_ID_list.append(trip_ID)
		sen_len.append(sen_length[arr[1]])
		sp_type.append(int(arr[-1]))
		arr[4] = arr[4].lower()
		IW.append(arr[4])
		try:
			IW_type.append(IW_type_dic[arr[4]][0])
			IW_cluster.append(PD.IWCluster[arr[4]])
			IW_dir.append(IW_dir_dic[arr[4]])
		except:
			if arr[4] != 'root':
				print('Warning in line '+str(line_count)+' : '+arr[4]+' is not in IW dictionary!')
			IW_type.append(99)
			IW_cluster.append(99)
			IW_dir.append(0)
		TF.append(int(arr[8].split('--')[0].split(':')[1]))
		DIR.append((1*(arr[9]=='ab') + 2*(arr[9]=='ba')))
		Trip = np.array([int(arr[5]) ,int(arr[6]) ,int(arr[7]) ])
		try:
			positions = np.vstack([ positions, Trip ])
		except:
			positions = Trip
		Prot_type.append(1*(arr[8].split('--')[-1]=='GENE-Y'))
		#------------------------------------------------------
		# Begin generate features in 2009 paper
		try:
			Interactor.append(IW_dic[arr[4]])
		except:
			IW_dic[arr[4]] = IW_count
			IW_count += 1
			Interactor.append(IW_dic[arr[4]])


		if Trip.argsort()[0]==2:
			Order.append(2)
		elif Trip.argsort()[1]==2:
			Order.append(1)
		else:
			Order.append(3)
		Trip = Trip[Trip.argsort()]
		D1.append(Trip[1] - Trip[0])
		D2.append(Trip[2] - Trip[1])
		if ',' in sen_sentence[trip_ID][Trip[0]:Trip[1]] and ',' in sen_sentence[trip_ID][Trip[1]:Trip[2]]:
			Comma.append(4)
		elif ',' in sen_sentence[trip_ID][Trip[0]:Trip[1]] and not ',' in sen_sentence[trip_ID][Trip[1]:Trip[2]]:
			Comma.append(2)
		elif ',' in sen_sentence[trip_ID][Trip[1]:Trip[2]] and not ',' in sen_sentence[trip_ID][Trip[0]:Trip[1]]:
			Comma.append(3)
		else:
			Comma.append(1)	

		if len(set(['not','incapable','unable']) & set(sen_sentence[trip_ID][Trip[0]:Trip[2]])) > 0:
			Not.append(2)
		else:
			Not.append(1)
		
		breaker_words = ['where', 'when', 'what', 'why', 'how', 'as', 'though', 'although', 'because', 'so', 'therefore', 'hence', 'since', 'wherein', 'whereas', 'whereby']
		if len(set(breaker_words) & set(sen_sentence[trip_ID][Trip[0]:Trip[2]])) > 0:
			Breaker.append(2)
		else:
			Breaker.append(1)
		
		if Trip[0]<3:
			prev3 = 0
		else:
			prev3 = Trip[0]-3
		if len(set(['whether','if']) & set(sen_sentence[trip_ID][prev3:Trip[0]])) > 0:
			Cdtnal.append(2)
		else:
			Cdtnal.append(1)
		
		Prep_words = ['with', 'of', 'by', 'multiple', 'to', 'between', 'via', 'through', 'in', 'for', 'on', 'within', 'during', 'from', 'without', 'at', 'under', 'among', 'after']
		if (int(arr[7])+3) > sen_length[trip_ID]:
			post3 = sen_length[trip_ID]
		else:
			post3 = int(arr[7]) + 4
		Prep_tmp = [i for i in sen_sentence[trip_ID][int(arr[7]):post3] if i in Prep_words]
		try:
			#tmp = Prep_dic['_'.join(set(Prep_words) & set(sen_sentence[trip_ID][int(arr[7]):post3]))]
			Prep.append(Prep_list.index('_'.join(Prep_tmp)))
		except:
			#Prep_dic['_'.join(set(Prep_words) & set(sen_sentence[trip_ID][int(arr[7]):post3]))] = Prep_count
			#Prep_count = Prep_count + 1
			#tmp = Prep_dic['_'.join(set(Prep_words) & set(sen_sentence[trip_ID][int(arr[7]):post3]))]
			Prep_list.append('_'.join(Prep_tmp))
			Prep.append(Prep_list.index('_'.join(Prep_tmp)))
		#Prep.append(tmp)
		
		if 'which' in sen_sentence[trip_ID][Trip[0]:Trip[2]]:
			Which.append(2)
		else:
			Which.append(1)
		
		if 'but' in sen_sentence[trip_ID][Trip[0]:Trip[2]]:
			But.append(2)
		else:
			But.append(1)
		
		IW_in_trips = [IW_type_dic[i][1] for i in sen_sentence[trip_ID][prev3:(Trip[2]+4)] if i in IW_type_dic.keys()]
		NofIW_len = len(IW_in_trips)
		if NofIW_len > 1:
			NofIW.append(2)
		else:
			NofIW.append(1)
		
		# end of generating features of 2009 paper
		#----------------------------------------------------
		prot_pos = [int(arr[5]),int(arr[6])]
		prot_pos.sort()
		if abs(prot_pos[1]-prot_pos[0])==1 or ( (abs(prot_pos[1]-prot_pos[0])==2) and (sen_sentence[trip_ID][prot_pos[0]+1]==',') ):
			isAdjacent.append(1)
		else:
			isAdjacent.append(0)
		
		for midx,m in enumerate([5,6]):
			tmp_arr = np.zeros(14,dtype = int)
			pre_Words = sen_sentence[trip_ID][max(0,int(arr[m])-3):int(arr[m])]
			pos_Words = sen_sentence[trip_ID][(int(arr[m])+1):(int(arr[m])+4)]
			tmp_arr[midx*5+0] = 1*(('-LSB-' in pre_Words or '-LRB-' in pre_Words) and ('-RSB-' in pos_Words or '-RRB-' in pos_Words))
			tmp_arr[midx*5+1] = 1*(len([i for i in (pre_Words+pos_Words) if 'produc' in i.lower()])>0)
			tmp_arr[midx*5+2] = 1*(len([i for i in (pre_Words+pos_Words) if 'pathway' in i.lower()])>0)
			tmp_arr[midx*5+3] = 1*(len([i for i in (pre_Words+pos_Words) if 'generat' in i.lower()])>0)
			tmp_arr[midx*5+4] = 1*(len([i for i in (pre_Words+pos_Words) if 'synthe' in i.lower()])>0)
			tmp_arr[midx*5+5] = 1*(len([i for i in (pre_Words+pos_Words) if 'substrate' in i.lower()])>0)
			tmp_arr[midx*5+6] = 1*(len([i for i in (pre_Words+pos_Words) if 'transport' in i.lower()])>0)
		try:
			isProduct = np.vstack([isProduct,tmp_arr])	
		except:
			isProduct = tmp_arr		

		if len([i for i in sen_sentence[trip_ID][prev3:(Trip[2]+4)]  if 'significant' in i.lower()]) > 0:
			Significant.append(1)
		else:
			Significant.append(0)
		
		if sen_sentence[trip_ID][int(arr[5])+1] == '-':
			p1_minus.append(1)
		else:
			p1_minus.append(0)
		if sen_sentence[trip_ID][int(arr[6])+1] == '-':
			p2_minus.append(1)
		else:
			p2_minus.append(0)
 
		tmp_arr = np.zeros(10,dtype=int) #type of iw included in triplet region
		if len(IW_in_trips) > 0:
			tmp_arr[np.unique(IW_in_trips)-1] = 1
		try:
			IW_CPR = np.vstack([IW_CPR,tmp_arr])
		except:
			IW_CPR = tmp_arr
		
		tmp_arr = np.zeros(10,dtype=int) # number of each type iw in the sentence
		tmp_iw1 = np.zeros([3,10],dtype=int) # # is each type of iw in {prot1-1,prot1+1}, {prot1-2,prot1+2}, {prot1-3,prot1+3}
		tmp_iw2 = np.zeros([3,10],dtype=int) # # is each type of iw in {prot1-1,prot1+1}, {prot1-2,prot1+2}, {prot1-3,prot1+3}
		tmp_iw3 = np.zeros(10,dtype=int) # is LUNGCPR before or after each type of iw  
		for ipos, i in enumerate(sen_sentence[trip_ID]):
			IWLIST = list(IW_type_dic.keys())
			IWLIST.sort()
			if i in IWLIST:
				tmp_arr[IW_type_dic[i][1]-1] = tmp_arr[IW_type_dic[i][1]-1] + 1
				for ii in range(1,4): 
					if ipos == (int(arr[5])-ii) or ipos == (int(arr[5])+ii): tmp_iw1[ii-1,IW_type_dic[i][1]-1] = 1
					if ipos == (int(arr[6])-ii) or ipos == (int(arr[6])+ii): tmp_iw2[ii-1,IW_type_dic[i][1]-1] = 1
				if len([ii for ii in sen_sentence[trip_ID][(ipos-3):(ipos+4)] if 'LUNGCPT' in ii]) > 0:
					tmp_iw3[IW_type_dic[i][1]-1] = 1
		tmp_iw1 = tmp_iw1.reshape(30)
		tmp_iw2 = tmp_iw2.reshape(30)
		try:
			nIW_CPR = np.vstack([nIW_CPR,tmp_arr])
			IW1_CPR = np.vstack([IW1_CPR,tmp_iw1])
			IW2_CPR = np.vstack([IW2_CPR,tmp_iw2])
			IW3_CPR = np.vstack([IW3_CPR,tmp_iw3])
		except:
			nIW_CPR = tmp_arr
			IW1_CPR = tmp_iw1
			IW2_CPR = tmp_iw2
			IW3_CPR = tmp_iw3

		NofP_len = len([i for i in sen_sentence[trip_ID][prev3:(Trip[2]+4)] if 'LUNGCPT' in i])
		NofP.append(NofP_len)		
		if NofP_len > 2:
			HofP.append(2)
		else:
			HofP.append(1)

		count = 0
		Steps = np.zeros([1,3],dtype=int)
	else:
		if len(arr) == 1 and 'NA' in arr[0]:    # no shortest path between this start node and goal node
			print(trip_ID+' has no shortest path '+str(count+1))
			Steps[0,count] = 100
		else:
			Steps[0,count] = (len(arr)+1)/2 -1
		
		#----------------------------------------------------
		# Begin generating binary type dependencies features
		tmp = np.zeros([len(td_list)], dtype = int)
		if len(arr) > 1:
			for i, ii in enumerate(arr[1::2]):	# arr[1::2] is a list of type dependencies in this shortest path
				tmp[td_list.index(ii)] = 1
		try:
			TD_SP[count] = np.vstack([TD_SP[count], tmp])
		except:
			TD_SP.append(tmp)
		count += 1

	if count==3:
		try:
			steps = np.vstack([steps,Steps])
		except:
			steps = Steps


n_cases = len(TF)
TF = np.array(TF,dtype=int)
TF[TF==0] = 10  # note: there is one pair labeled CPR:0 UNDIFINED, we replace it with CPR:10
np.savetxt(prefix+'_labels_triplet.txt', TF, delimiter=',', fmt='%d')
OUT = open(prefix+'_tripletID.txt', 'w')
for i in trip_ID_list:
	OUT.write(i+'\n')
OUT.close()

Feature2017_names = ['sp_type','SenLen','steps_sp1','steps_sp2','steps_sp3','pos_p1','pos_p2','pos_iw','Significant','p1_minus','p2_minus',
                     'direction','cluster_iw','IW_type','NumberOfProteins','isHighNofP','IW_direction','TypeOfProtein','IW1','IW2','IW3','IW4','IW5','IW6','IW7','IW8','IW9','IW10']
X = np.zeros([n_cases,len(Feature2017_names)])
X[:,1] = np.array(sen_len)
X[:,0] = np.array(sp_type)
X[:,2:5] = steps
X[:,5:8] = positions
X[:,8] = np.array(Significant)
X[:,9] = np.array(p1_minus)
X[:,10] = np.array(p2_minus)
X[:,11] = np.array(DIR)
X[:,12] = np.array(IW_type)
X[:,13] = np.array(IW_cluster)
X[:,14] = np.array(NofP)
X[:,15] = np.array(HofP)
X[:,16] = np.array(IW_dir)
X[:,17] = np.array(Prot_type)
X[:,18:] = IW_CPR

#added on 9/19
Feature2017_names = Feature2017_names+ ['nIW1','nIW2','nIW3','nIW4','nIW5','nIW6','nIW7','nIW8','nIW9','nIW10']
X = np.hstack([X,nIW_CPR])

#added on 9/20
for rr in range(3): 
	for r in range(10): 
		Feature2017_names = Feature2017_names+ ['p1IW'+str(r+1)+'_'+str(rr+1)]
X = np.hstack([X,IW1_CPR])
for r in range(10): 
	for rr in range(3): 
		Feature2017_names = Feature2017_names+ ['p2IW'+str(r+1)+'_'+str(rr+1)]
X = np.hstack([X,IW2_CPR])
Feature2017_names = Feature2017_names+ ['p3IW1','p3IW2','p3IW3','p3IW4','p3IW5','p3IW6','p3IW7','p3IW8','p3IW9','p3IW10']
X = np.hstack([X,IW3_CPR])
#added on 9/20
Feature2017_names = Feature2017_names+['isBracket1','isBracket2','isProduct1','isProduct2','isPathway1','isPathway2',
                                       'isGenerate1','isGenerate2','isSynthetic1','isSynthetic2','isAdjacent']
X = np.hstack([X,isProduct])
X = np.hstack([X,np.array(isAdjacent).reshape([n_cases,1])])

Feature2009_names = ['Interactor','D1','D2','Order','Comma','Not','Breaker','Conditional','Prep','Which','But','NumberOfInteractors']
Features2009 = np.zeros([n_cases,len(Feature2009_names)])
Features2009[:,0] = Interactor
Features2009[:,1] = D1
Features2009[:,2] = D2
Features2009[:,3] = Order
Features2009[:,4] = Comma
Features2009[:,5] = Not
Features2009[:,6] = Breaker
Features2009[:,7] = Cdtnal
Features2009[:,8] = Prep
Features2009[:,9] = Which
Features2009[:,10] = But
Features2009[:,11] = NofIW

Feature_names = Feature2017_names + Feature2009_names
Features = np.hstack([X,Features2009])
print(Features.shape, len(Feature_names))
for i in range(3):
	for k in td_list:
		Feature_names.append(k+'_'+str(i+1))
	Features = np.hstack([Features, TD_SP[i]])

np.savetxt(prefix+'_Features_triplet.txt',Features, delimiter = ',', fmt='%d')
OUT = open(prefix+'_FeatureNames_triplet.txt', 'w')
for i in Feature_names:
	OUT.write(i+'\n')
OUT.close()

