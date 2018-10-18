import os, sys, getopt
import numpy as np

argv = sys.argv[1:]
try:
        opts, args = getopt.getopt(argv,"hp:",["prefix="])
except getopt.GetoptError:
        print('usage: Dep2Graph.py -p <prefix>')
        sys.exit(2)

for opt, arg in opts:
        if opt == '-h':
                print('usage: Dep2Graph.py -p <prefix>')
                sys.exit()
        elif opt in ("-p","--prefix"):
                prefix = arg


os.system('rm '+prefix+'_nnparsed_graph.txt')


#Step2: Dep to graph
tmp_IN = open(prefix+'_triplets.txt')
pID = []
for i in tmp_IN:
        pID.append(i.split('|')[1].split('\t')[0])

IN = open(prefix+'_sentences_sen.dep', 'r')
OUT = open(prefix+'_nnparsed_graph.txt', 'w')


LINES = IN.readlines()
trip_ID = 0
ID = ''
ID_count = 0
for i, line in enumerate(LINES):
    if not line == '\n':
        tmp = line[line.find('(')+1:line.rfind(')')].split(', ')
        dep = line[0:line.find('(')]
        n1 = tmp[0][0:tmp[0].rfind('-')]
        ind1 = tmp[0][tmp[0].rfind('-')+1:]
        n2 = tmp[1][0:tmp[1].rfind('-')]
        ind2 = tmp[1][tmp[1].rfind('-')+1:]

    if n2 == '###':
        if pID[ID_count] != ID:
            trip_ID = 0
        ID = pID[ID_count]
        ID_count += 1

    #if len(set([n1,n2]) & set(ID.split('.'))) > 0 or '@' in [n1, n2]:
    if len(set([n1,n2]) & set(ID.split('.'))) > 0:        
        n1 = ''
        n2 = ''
        continue
    elif line == '\n':
        try:
            if not '###' in LINES[i+1]:
                #print(ID,trip_ID)
                OUT.write('#\t' + ID + '_' + str(trip_ID) + '\n')
                trip_ID += 1

        except:
                print('End of file.')
    else:
        OUT.write(dep + '\t' + n1 + '\t' + ind1 + '\t' + n2 + '\t' + ind2 + '\n')


OUT.close()

