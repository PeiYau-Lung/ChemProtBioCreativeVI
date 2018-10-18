
# This program generates triplets from original dataset with ID and abstracts, based on the relations file
# In this version, we firstly replace all gene names in abstracts with LUNGCPT1, LUNGCPT2, ... and so on.

import numpy as np
import sys, getopt
from nltk.tokenize import sent_tokenize, word_tokenize

 
argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv,"ha:e:r:p:",["abstract=","entities=","relation=s","prefix="])
except getopt.GetoptError:
	print('usage: Abs2Triplets.py -a <AbstractFile> -e <EntitiesFile> -r <RelationsFile> -p <profix>')
	sys.exit(2)
 
for opt, arg in opts:
	if opt == '-h':
		print('usage: Abs2Triplets.py -a <AbstractFile> -e <EntitiesFile> -r <RelationsFile> -p <profix>')
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
	new_str = String.replace('\xe2\x88\xbc','~').replace('\xe2\x86\x93','>').replace('\xe2\x88\x9e','i').replace('\xe2\x82\x83',' ').replace('\xe2\x80\x82',' ').replace('\xe2\x80\x89',' ').replace('\xe2\x89\xa5',">").replace('\xe2\x89\xa4',"<").replace('\xc2\xb1',"+").replace('\xe2\x82\x81','1').replace('\xce\xb1','a').replace('\xc2\xb4',"'").replace('\xce\xb2','b').replace('\xc2\xa0',' ').replace('\xc2\xb5','u').replace('\xce\xcb','v').replace('\xce\xbc','u').replace('\xce\xba','k').replace('\xce\xb3','y').replace('\xce\xb4','d').replace('\xc2\xa9','c').replace('\xce\x94','d').replace('\xc3\x97','x').replace('\xcf\x80','p').replace('\xc3\xb6','o').replace('\xcf\x89','w').replace('\xc2\xb0','o').replace('\xc2\xb7','*').replace('\xc3\xbc','u')
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
Abs_dict = {}	#key:ID value:abstract
for i in ABS:
	arr = i.strip().split('\t')
	if len(arr) < 3:
		abs_tmp = arr[1]
	else:
		abs_tmp = arr[1]+'\t'+arr[2]

	Abs_dict[arr[0]] = ReplaceUni(abs_tmp)

#------------------------------
# getting all chemical compound and gene names and their offsets in the corresponding abstracts
#-----------------------------
ENT = open(entitiesfile, 'r')
ENT_dict = {}	#key:ID value:a dictionary of {LUNGCPT* : [type, start, end, name]}. LUNGCP is added before T i order not to be duplicated with other words 
OUT_PN = open(prefix+'_ProteinNames.txt','w')
Protein_name_list = []
Protein_type_list = []
for i in ENT:
	arr = i.split('\n')[0].split('\t')
	arr[-1] = ReplaceUni(arr[-1])
	try:
		ENT_dict[arr[0]]
	except:
		ENT_dict[arr[0]] = {}
	# if Ti < T10, replaced it with T0i e.g. T3 -> T03
	if int(arr[1][1:]) < 10:
		tmp_ent = 'LUNGCPT0' + arr[1][1:]
	else:
		tmp_ent = 'LUNGCP'+arr[1]
	ENT_dict[arr[0]][tmp_ent] = arr[2:]
	if not arr[-1].replace('\n','') in Protein_name_list:
		Protein_name_list.append(ReplaceUni(arr[-1].replace('\n','')))
		Protein_type_list.append(arr[2])
for idx, i in enumerate(Protein_name_list):
	OUT_PN.write(i+'\t'+Protein_type_list[idx]+'\n')
OUT_PN.close()

#-------------------------------
# extracting triplets
#-------------------------------
REL = open(relationsfile, 'r') 
OUT = open(prefix+"_triplets.txt", 'w')
OUT_label = open(prefix+"_pairs_label.txt", 'w')

pair_id_list = []
absid = ''
for line_id, line in enumerate(REL):
	arr = line.replace('\n','').split('\t')
	if arr[0] == absid:
		pair_count += 1
	else:
		pair_count = 0
		absid = arr[0]

	if True:
		if int(arr[-2].split(':')[-1][1:]) < 10:
			e1 = 'LUNGCPT0' + arr[-2].split(':')[-1][1:]
		else:
			e1 = 'LUNGCP' + arr[-2].split(':')[-1]
		if int(arr[-1].split(':')[-1][1:]) < 10:
			e2 = 'LUNGCPT0' + arr[-1].split(':')[-1][1:]
		else:
			e2 = 'LUNGCP' + arr[-1].split(':')[-1]
		pair_id = absid + '.pair' + str(pair_count)
		if not pair_id in pair_id_list:
			OUT_label.write(pair_id+'\t'+arr[1]+'\t'+arr[2]+'\t'+arr[3]+'\n')
			pair_id_list.append(pair_id)
		
		# dealing with sen for this pair
		sen_forpair = Abs_dict[absid]
		try:
			e1_offset = ['-'.join(ENT_dict[absid][e1][1:3])]
		except:
			print(e1,arr[-2])
			sys.exit()
		try:
			e2_offset = ['-'.join(ENT_dict[absid][e2][1:3])]
		except:
			print(e2,arr[-1])
			sys.exit()
		
		#-------------------------------------------------------------------------------------------
		if len(e1_offset) == 1 and len(e2_offset)==1:	# neither one is separated
			e11, e12 = [int(i) for i in e1_offset[0].split('-')]
			e21, e22 = [int(i) for i in e2_offset[0].split('-')]
			 
			if e11 < e21 and e12 < e21:	#no overlapping, 1st pro is left-hand side to 2nd pro
				sen_forpair = sen_forpair[:e21] + e2 + sen_forpair[e22:]
				sen_forpair = sen_forpair[:e11] + e1 + sen_forpair[e12:]
			elif e21 < e11 and e22 < e11:     #no overlapping, 2st pro is left-hand side to 1nd pro
				sen_forpair = sen_forpair[:e11] + e1 + sen_forpair[e12:]
				sen_forpair = sen_forpair[:e21] + e2 + sen_forpair[e22:]
			elif e11 <= e21 and e21 <= e12:	#overlapping, 1st pro first and next 2nd pro
				sen_forpair = sen_forpair[:e11] + e1 + ' ' + e2 + sen_forpair[max(e12,e22):]
			elif e21 <= e11 and e11 <= e22:	#overlapping, 2nd pro first and next 1st pro
				sen_forpair = sen_forpair[:e21] + e2 + ' ' + e1 + sen_forpair[max(e12,e22):]
		#-------------------------------------------------------------------------------------------
		
		try:
			tmp_sen_forpair = sent_tokenize(sen_forpair.replace('\t',' '))
		except:
			print(absid)	
		tmp_sen_forpair = sent_tokenize(sen_forpair.replace('\t',' '))	# replace the tab-delimiter between title and abstract with space, and split into sentences
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
 	
		if len([i for i in sen_forpair if e1 in i and e2 in i]) != 1:
			print('Two entities are in different sentences. Please check!')
			print(e1, e2)
			for i, j in enumerate(sen_forpair):
				print(i,j)
				if e1 in j and e2 in j: print(j)

			sys.exit()
		
		sen_forpair = [i for i in sen_forpair if e1 in i and e2 in i][0]
		#---------------------------------------------------------------------
		# replace other protein names in this sentence with their entity ID
		#sen_forpair = sen_forpair.replace(e1, '***').replace(e2,'~~~')
		sen_forpair = sen_forpair.replace(e1, '***').replace(e2,'~~~').replace('down-regulat','downregulat').replace('Down-regulat','Downregulat').replace('up-regulat','upregulat').replace('Up-regulat','Upregulat')
		ent_id = list(ENT_dict[absid].keys())
		ent_len = np.array([len(ENT_dict[absid][i][-1]) for i in ent_id])
		ent_len_sortidx = np.argsort(-ent_len) # index of sorted ent length from largest to smallest 		
		for m in ent_len_sortidx:
			i = ent_id[m]
			if i == e1 or i == e2:
				continue
			else:
				if ENT_dict[absid][i][-1] in sen_forpair and len(ENT_dict[absid][i][-1]) > 1 and ENT_dict[absid][i][-1] != 'CPT':
					sen_forpair = sen_forpair.replace(ENT_dict[absid][i][-1],i)
		sen_forpair = sen_forpair.replace('***', e1).replace('~~~', e2)
		#----------------------------------------------------------------------
		sen_forpair = sen_forpair.replace('/',' / ').replace('(',' ( ').replace(')',' ) ').replace(',',' ,').replace(':',' : ').replace('-',' - ').replace('+',' + ')
		#sen_arr = sen_forpair.split()
		sen_arr = word_tokenize(sen_forpair)
		#-----------------------------
		# manually correct some annotation problem
		#----------------------------
		#if pair_id == '22349823.p2':
		#	sen_arr[sen_arr.index('LUNGCPT1790')] = 'LUNGCPT1'
						

		IWs = [(j,i) for i,j in enumerate(sen_arr) if j.lower() in IW_list and j!='']
		if len(IWs) == 0:
			IWs = [('IW_NA',-1)]	
	
		p1 = [j for j, k in enumerate(sen_arr) if e1 == k]
		if len(p1) == 0:
			p1 = [j for j, k in enumerate(sen_arr) if e1 in k]
		p2 = [j for j, k in enumerate(sen_arr) if e2 == k]
		if len(p2) == 0:
			p2 = [j for j, k in enumerate(sen_arr) if e2 in k]
		
		if len(p2) > len(p1):
			p2 = [j for j in p2 if j not in p1]
		elif len(p1) > len(p2):
			p1 = [j for j in p1 if j not in p2]
		if not (len(p1)*len(p2)) ==1:
			print(pair_id)
			print(sen_forpair)
			print(e1,p1,e2,p2,sen_arr)
			sys.exit()
		#print(p1,p2,IWs,sen_arr)
		#sys.exit()
		if len(e1) > len(e2):
			sen_arr[p1[0]] = sen_arr[p1[0]].replace(e1,'PROT1')
			sen_arr[p2[0]] = sen_arr[p2[0]].replace(e2,'PROT2')
		else:
			sen_arr[p2[0]] = sen_arr[p2[0]].replace(e2,'PROT2')
			sen_arr[p1[0]] = sen_arr[p1[0]].replace(e1,'PROT1')
		sen_towrite = ' '.join(sen_arr)
		
		for i in IWs:
			tmp, piw = i
			#if pair_id == 'BioInfer.d835.s0.p9':
			#	print(p1,e1,p2,e2,sen_forpair,sen_arr)
			if tmp == 'IW_NA':
				print('Warning! '+pair_id+' has no interaction word.')
			
			OUT.write('--'.join(arr[1:4])+'--'+ENT_dict[arr[0]][e2][0]+'\t'+absid+'|'+pair_id+'\tPROT1_'+ENT_dict[absid][e1][-1]+'\tPROT2_'+ENT_dict[absid][e2][-1]+'\t'+tmp.lower()+'\t'+str(p1[0]+1)+'\t'+str(p2[0]+1)+'\t'+str(piw+1)+' | '+sen_towrite+'\n')





OUT.close()
OUT_label.close()
OUT_PN.close()












		















 
