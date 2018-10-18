'''
This program generate tagged file and labled triplets file from corpus.

The output format for tagged file is 
ID\t|PROT1_pos1|PROT2_pos2\t|IW_posiw\tsentence

The output format for labeled triplets file is 
label\tID\tPROT1\tPROT2\tIW\tpos1\tpos2\tposiw\tdirection(NA)\tsentence 
'''


import os, sys, getopt
from nltk.tokenize import word_tokenize

argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(argv,"hs:c:p:",["senfile=","corfile=","prefix="])
except getopt.GetoptError:
	print('usage: TagTriplets.py -s <sentencefile> -c <corpusfile> -p <prefix>')
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print('usage: TagTriplets.py -s <sentencefile> -c <corpusfile> -p <prefix>')
		sys.exit()
	elif opt in ("-s", "--senfile"):
		senfile = arg
	elif opt in ("-c", "--corfile"):
		corfile = arg
	elif opt in ("-p", "--prefix"):
		prefix = arg


COR = open(corfile,'r')
SEN = open(senfile,'r')
OUT = open(prefix+'_tagged.txt','w')
LF = open(prefix+'_trip_label.txt','w')

for i, line in enumerate(SEN):
	arr = line.strip().split('\t')
	ID = arr[0]
	#sentence = arr[1].strip().split(' ')
	sentence = word_tokenize(arr[1].strip())
	trip = COR.readline()
	IW = trip.strip().split('|')[1].strip().split('\t')[3]
	label = trip.strip().split('|')[0].strip().split('\t')[0]
	
	#pos1 = sentence.index('PROT1')
	#pos2 = sentence.index('PROT2')
	pos1 = [j for j, w in enumerate(sentence) if 'PROT1' in w][0] + 1
	pos2 = [j for j, w in enumerate(sentence) if 'PROT2' in w][0] + 1
	if IW in ['IW_NA','iw_na']:
		posiw = 0
	else:
		posiw = [j+1 for j, w in enumerate(sentence) if IW == w.lower()]
		if len(posiw) == 0 :
			print('Wrong! '+ str(i)+ ': '+trip+'\n')
			sys.exit()
		if len(posiw) > 1:
			posiw_cor = int(trip.strip().split('|')[1].strip().split('\t')[6])
			tmp = [abs(j-posiw_cor) for j in posiw]
			posiw = posiw[tmp.index(min(tmp))]
		else:
			posiw = posiw[0]
	OUT.write(ID + '\t|PROT1_' + str(pos1) + '|PROT2_' + str(pos2) + '\t|' + IW.lower() + '_' + str(posiw) + '\t' + ' '.join(sentence) + '\n')
	LF.write(label+'\t'+ID+'\tPROT1\tPROT2\t'+IW.lower()+'\t'+ str(pos1)+'\t'+ str(pos2)+'\t'+ str(posiw)+'\tNA\t'+' '.join(sentence)+'\n')

OUT.close()
LF.close()





