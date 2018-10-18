# This program uses entities files to generate all possible relation pairs.
# Two entities in a pair should be in the same sentence.
# If a pair is not in the relation file, this pair should be CPR10.

import numpy as np
import sys, getopt
from nltk.tokenize import sent_tokenize, word_tokenize


argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv,"ha:e:r:p:",["abstract=","entities=","relation=s","prefix="])
except getopt.GetoptError:
	print('usage: Ent2relation.py -a <AbstractFile> -e <EntitiesFile> -r <RelationsFile> -p <prefix>')
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print('usage: Ent2Relation.py -a <AbstractFile> -e <EntitiesFile> -r <RelationsFile> -p <prefix>')
		sys.exit()
	elif opt in ("-a", "--abstract"):
		abstractfile = arg
	elif opt in ("-p", "--prefix"):
		prefix = arg
	elif opt in ('-e','--entities'):
		entitiesfile = arg
	elif opt in ('-r', '--relation'):
		relationsfile = arg

#------------------------------------
# Defining a function that replaces unicode to letter
#-----------------------------------
def ReplaceUni(String):
	new_str = String.replace('\xe2\x88\xbc','~').replace('\xe2\x88\x9e','i').replace('\xe2\x86\x93','>').replace('\xe2\x82\x83',' ').replace('\xe2\x80\x82',' ').replace('\xe2\x80\x89',' ').replace('\xe2\x89\xa5',">").replace('\xe2\x89\xa4',"<").replace('\xc2\xb1',"+").replace('\xe2\x82\x81','1').replace('\xce\xb1','a').replace('\xc2\xb4',"'").replace('\xce\xb2','b').replace('\xc2\xa0',' ').replace('\xc2\xb5','u').replace('\xce\xcb','v').replace('\xce\xbc','u').replace('\xce\xba','k').replace('\xce\xb3','y').replace('\xce\xb4','d').replace('\xc2\xa9','c').replace('\xce\x94','d').replace('\xc3\x97','x').replace('\xcf\x80','p').replace('\xc3\xb6','o').replace('\xcf\x89','w').replace('\xc2\xb0','o').replace('\xc2\xb7','*').replace('\xc3\xbc','u')
	return new_str

#-------------------------------
# geting all interation words
#------------------------------
IW = open('./src/IW_CPR_09_2017.txt','r')
IW_list = []
for line in IW:
	if line.startswith('#'):
		continue
	elif line == '\n':
		continue
	else:
		IW_list.append(line.strip().split('\t')[0])


#-------------------------------
# geting all abstracts
#------------------------------
ABS = open(abstractfile, 'r')
Abs_dict = {}   #key:ID value:abstract
for i in ABS:
	arr = i.strip().split('\t')
	if len(arr) < 3:
		abs_tmp = arr[1]
	else:
		abs_tmp = arr[1]+'\t'+arr[2]

	Abs_dict[arr[0]] = ReplaceUni(abs_tmp)


#-------------------------------
# geting all relations
#------------------------------
REL = {}
for i in np.unique(list(Abs_dict.keys())):
	REL[i] = {}
try:
	RF = open(relationsfile, 'r')
	for i, line in enumerate(RF):
		arr = line.split('\n')[0].split('\t')
		e1 = arr[4].split(':')[1]
		e2 = arr[5].split(':')[1]
		try:
			if REL[arr[0]][e1+'--'+e2] != arr[1] and arr[1] in ['CPR:3', 'CPR:4','CPR:5','CPR:6','CPR:9']:
				# some pairs in relation file are duplicated with different labels. We keep those pairs whose CPR are true
				REL[arr[0]][e1+'--'+e2] = arr[1]
		except:
			REL[arr[0]][e1+'--'+e2] = arr[1]			
				
except:
	print('no relation files. all pair will be labeled as CPR10.')


#------------------------------
# getting all chemical compound and gene names and their offsets in the corresponding abstracts
#-----------------------------
ENT = open(entitiesfile, 'r')
CHEMPROT = {}
OFF = {}
for i, line in enumerate(ENT):
	arr = line.split('\n')[0].split('\t')
	absid = arr[0]
	try:
		CHEMPROT[absid]
	except:
		CHEMPROT[absid] = {'chem' : [], 'prot' : []}
	
	if arr[2] == 'CHEMICAL':
		CHEMPROT[absid]['chem'].append(arr[1])
	else:
		CHEMPROT[absid]['prot'].append(arr[1])
	try:
		OFF[absid][arr[1]] = arr[3:]
	except:
		OFF[absid] = {arr[1] : arr[3:]}


#------------------------------
# generate relation pair
#-----------------------------
REL_OUT = open(prefix+'_ALLrelations.txt', 'w')
ABSID = np.unique(list(OFF.keys()))
for absid in ABSID:
	for i, chem_ent in enumerate(CHEMPROT[absid]['chem']):
		for j, prot_ent in enumerate(CHEMPROT[absid]['prot']):
			sen_forpair = Abs_dict[absid]
			#print(chem_ent, prot_ent)
			if int(chem_ent[1:]) < 10:
				e1 = 'LUNGCPT0' + chem_ent[1:]
			else:
				e1 = 'LUNGCP' + chem_ent
			if int(prot_ent[1:]) < 10:
				e2 = 'LUNGCPT0' + prot_ent[1:]
			else:
				e2 = 'LUNGCP' + prot_ent
			e1_offset = ['-'.join(OFF[absid][chem_ent][0:2])]
			e2_offset = ['-'.join(OFF[absid][prot_ent][0:2])]
			#-------------------------------------------------------------------------------------------
			if len(e1_offset) == 1 and len(e2_offset)==1:   # neither one is separated
				e11, e12 = [int(i) for i in e1_offset[0].split('-')]
				e21, e22 = [int(i) for i in e2_offset[0].split('-')]

				if e11 < e21 and e12 < e21:     #no overlapping, 1st pro is left-hand side to 2nd pro
					sen_forpair = sen_forpair[:e21] + e2 + sen_forpair[e22:]
					sen_forpair = sen_forpair[:e11] + e1 + sen_forpair[e12:]
				elif e21 < e11 and e22 < e11:     #no overlapping, 2st pro is left-hand side to 1nd pro
					sen_forpair = sen_forpair[:e11] + e1 + sen_forpair[e12:]
					sen_forpair = sen_forpair[:e21] + e2 + sen_forpair[e22:]
				elif e11 <= e21 and e21 <= e12: #overlapping, 1st pro first and next 2nd pro
					sen_forpair = sen_forpair[:e11] + e1 + ' ' + e2 + sen_forpair[max(e12,e22):]
				elif e21 <= e11 and e11 <= e22: #overlapping, 2nd pro first and next 1st pro
					sen_forpair = sen_forpair[:e21] + e2 + ' ' + e1 + sen_forpair[max(e12,e22):]
			#-------------------------------------------------------------------------------------------

			try:
				tmp_sen_forpair = sent_tokenize(sen_forpair.replace('\t',' '))
			except:
				print(absid)
			tmp_sen_forpair = sent_tokenize(sen_forpair.replace('\t',' '))  # replace the tab-delimiter between title and abstract with space, and split into sentences
			#--------------------------------------------
			#dealing with some bad separation from tokenizer
			#-------------------------------------------
			sen_forpair = [tmp_sen_forpair[0]]
			tmp_sen_forpair = tmp_sen_forpair[1:]
			while len(tmp_sen_forpair) > 0:
				cond1 = sen_forpair[-1][-4:] in ['e.g.','i.e.','i.m.','a.u.','p.o.','i.v.','i.p.']
				cond2 = sen_forpair[-1][-5:] in ['vivo.','p.o.)','i.p.)','Vmax.']
				cond3 = sen_forpair[-1][-6:] in ['i.c.v.']
				cond4 = sen_forpair[-1][-7:] in ['i.c.v.)']
				cond5 = sen_forpair[-1][-3:] in [')(.']
				if not cond1 and not cond2 and not cond3 and not cond4 and not cond5:
					sen_forpair.append(tmp_sen_forpair[0])
				else:
					sen_forpair[-1] = sen_forpair[-1] + ' ' + tmp_sen_forpair[0]
				tmp_sen_forpair.remove(tmp_sen_forpair[0])

			if len([i for i in sen_forpair if e1 in i and e2 in i]) != 1 or e1 in e2 or e2 in e1:
				aaa=1
				#print('Two entities are in different sentences. Please check!')
				#print(e1,e2,sen_forpair)
				#sys.exit()
			else:
				tmp_str = 'T' + str(int(e1[-2:])) + '--' + 'T' + str(int(e2[-2:]))
				try:
					CPR = REL[absid][tmp_str]
				except:
					CPR = 'CPR:10'
				ARG1 = 'T' + str(int(e1.split('LUNGCPT')[-1]))
				ARG2 = 'T' + str(int(e2.split('LUNGCPT')[-1]))
				if CPR in ['CPR:3', 'CPR:4','CPR:5','CPR:6','CPR:9']:
					NY = 'Y'
				else:
					NY = 'N'
				
				REL_OUT.write(absid+'\t'+CPR+'\t'+NY+'\tNA\tArg1:'+ARG1+'\tArg2:'+ARG2+'\n')
			


REL_OUT.close()






















