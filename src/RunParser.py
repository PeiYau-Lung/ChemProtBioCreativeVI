'''
This program runs the Standford parser. The input should be two columns, separated by \t.
The first is ID and the second is sentencs.
'''

import os, sys, getopt
import numpy as np

argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv,"hi:p:",["ifile=","prefix="])
except getopt.GetoptError:
	print('usage: RunParser.py -i <inputfile> -p <prefix>')
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print('usage: RunParser.py -i <inputfile> -p <prefix>')
		sys.exit()
	elif opt in ("-i", "--ifile"):
		inputfile = arg
	elif opt in ("-p","--prefix"):
		prefix = arg


#os.system('rm '+prefix+'_nnparsed_graph.txt')
os.system('rm '+prefix+'_sentences_sen.dep')

#Step1: seperate ID and sentences into two files; parsed all sentenses 
IN = open(inputfile,'r')
OUT_SEN = open(prefix+'_sentences_sen.txt','w')
for line in IN:
    arr = line.strip().split('\t')
    OUT_SEN.write('###'+arr[0]+' @\n')
    OUT_SEN.write(arr[1]+' @\n')
    
OUT_SEN.close()


os.system('java -Xmx15000m -cp "./src/stanford-corenlp-full-2017-06-09/*" edu.stanford.nlp.parser.nndep.DependencyParser -model edu/stanford/nlp/models/parser/nndep/english_UD.gz -sentenceDelimiter @ -textFile '+prefix+'_sentences_sen.txt -outFile '+prefix+'_sentences_sen.dep')
