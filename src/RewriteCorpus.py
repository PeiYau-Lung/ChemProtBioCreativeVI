'''
This program rewrite corpus like LLL_triplets.txt into [ID \t sentence] in each line.
There are ___, which are other protein names besides PROT1 and PROT2, in corpus.
We replace ___ with PROT3, PROT4...
'''

import sys, getopt


argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
except getopt.GetoptError:
	print('usage: RewriteCorpus.py -i <inputfile> -o <outputfile>')
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print('usage: RewriteCorpus.py -i <inputfile> -o <outputfile>')
		sys.exit()
	elif opt in ("-i", "--ifile"):
		inputfile = arg
	elif opt in ("-o", "--ofile"):
		outputfile = arg
	
IN = open(inputfile, 'r')
OUT = open(outputfile, 'w')

trip_count = 0
sen_ID = []
for line in IN:
	arr = line.strip().split('|')
	if arr[1].strip().split('\t')[0] != sen_ID:
		trip_count = 0
	sen_ID = arr[1].strip().split('\t')[0]
	#trip_ID = sen_ID + '_' + str(trip_count)
	trip_ID = sen_ID
	trip_count += 1
	
	sen = arr[-1].strip().replace('-',' - ').replace('(',' ( ').replace(')',' ) ').replace('/',' / ').split(' ')
	p_count = 3
	for i, word in enumerate(sen):
		if word == '':
			continue
		if word in '_______________________':
			sen[i] = 'PROT'+str(p_count)
			p_count += 1
					
	sentence = ' '.join(sen).replace(' - ','-').replace(' ( ','(').replace(' ) ',')')

	OUT.write(trip_ID + '\t' + sentence + '\n')

OUT.close()


