#!/usr/bin/python

# this program takes two files as input: a file with bio-enity names and relationship words tagged (PICAD_tagged.txt)
# and a file with dependency graph from stanford parser (PICAD_nnparsed_graph.txt)

import sys, os, getopt
sys.path.append("./src/")
from astar import *
from graph import Graph
from sp import *
import numpy as np
#import math
argv = sys.argv[1:]

def sldist(a,b):
	return(1.0)

try:
	opts, args = getopt.getopt(argv,"hg:t:l:p:",["gfile=","tfile=","lfile=","prefix="])
except getopt.GetoptError:
	print('usage: shortestPath.py -g <graphfile> -t <taggedfile> -l <LabeledTripFile> -p <prefix>')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('usage: shortestPath.py -g <graphfile> -t <taggedfile> -l <LabeledTripFile> -p <prefix>')
		sys.exit()
	elif opt in ("-g", "--gfile"):
		graphfile = arg
	elif opt in ("-t", "--tfile"):
		taggedfile = arg
	elif opt in ("-l", "--lfile"):
		labelfile = arg
	elif opt in ("-p", "--prefix"):
		prefix = arg

# graph file
line_total = sum(1 for i in open(graphfile))
GF = open(graphfile, 'r')

# tagged file
TF = open(taggedfile, 'r')

# label information
LF = open(labelfile, 'r')

# output file
OUT = open(prefix+"_shortest_path.txt",'w')

tagDic = {}

for line in TF:
	arr = line.strip().split('\t')
	tagDic[arr[0]] = arr[1] + "---" + arr[2]

tripDic = {}
Cnt = 0
for line in LF:
	arr = line.strip().split('\t')
	label = arr[0]
	direction = arr[8]
	sent = arr[9]
	#sentID = arr[1][0:arr[1].rfind('_')]
	sentID = arr[1]
	key1 = sentID + '_' + arr[5] + '_' + arr[6] + '_' + arr[7].lower()
	key2 = sentID + '_' + arr[6] + '_' + arr[5] + '_' + arr[7].lower()
	value = label + "\t" + direction
	tripDic[key1] = value
	tripDic[key2] = value
	if label == 't':
		Cnt = Cnt + 1
		
#print("Number of true triplets is ", Cnt)

Count = 0
edgeDic = {}
TripCnt = 0
line_count = 0
for line in GF:
	line_count += 1
	arr = line.strip().split('\t')
	if arr[0] == '#' and Count == 0:
		ID = arr[1]
		g = Graph()
		spg = SPG()
		Count = Count + 1		
	elif arr[0] == '#' or line_count == line_total:
		if ID in tagDic:
			tag = tagDic[ID]
		else:
			print("ID not found ", ID)
			ID = arr[1]
			g = Graph()
			continue
		arr2 = tag.split("---")
		names = arr2[0].split('|')[1:]
		iws = arr2[1].split('|')[1:]
		
		if line_count == line_total: #if now is the last line, update it into graph
			edgeDic[(int(arr[2]), int(arr[4]))] = line
			edgeDic[(int(arr[4]), int(arr[2]))] = line
			g.add_node(int(arr[2]), arr[1])
			g.add_node(int(arr[4]), arr[3])
			g.add_edge(int(arr[2]), int(arr[4]), 1, arr[0])
			g.add_edge(int(arr[4]), int(arr[2]), 1, arr[0])	#add by PYL, 2017/09/14	
		
		for iw in iws:
			iwpos = int(iw.split('_')[-1])
			#for i in range(len(names)-2): # bug. should be len(names)-1
			for i in range(len(names)-1):
				pos1 = int(names[i].split('_')[-1])
				for j in range(i+1, len(names)):
					pos2 = int(names[j].split('_')[-1])
					#print ID, iwpos, pos1, pos2
					#print ID, names[i].split('_')[0], names[j].split('_')[0], iw.split('_')[0]
					spg.clear_path()
					sp1 = shortest_path(g, pos1, iwpos, sldist)
					spg.add_path(sp1)
					sp2 = shortest_path(g, iwpos, pos2, sldist)
					spg.add_path(sp2)
					sp3 = shortest_path(g, pos1, pos2, sldist)
					spg.add_path(sp3)
					spg.computeType()
					#sp = sp1+sp2+sp3
					#edgeSet = set()
					#for k in range(len(sp)-1):
					#	if sp[k] == sp[k+1]:
					#		continue
					#	edgeSet.add(edgeDic[(sp[k],sp[k+1])])
					key = ID + '_' + str(pos1) + '_' + str(pos2) + '_' + str(iwpos)
					if key not in tripDic:
						print("key not in tripDic: ", key)
						value = "NA\tNA"
					else:
						value = tripDic[key]
					
					try:
						OUT.write('#\t'+ID+'\t'+g.nodeType[pos1]+'\t'+g.nodeType[pos2]+'\t'+g.nodeType[iwpos])
						OUT.write('\t'+str(pos1)+'\t'+str(pos2)+'\t'+str(iwpos)+'\t'+value+'\t'+str(spg.type)+'\n')
					except:
						print(ID)
						print(g.nodeType)
						print(g.nodeType[pos1], g.nodeType[pos2], g.nodeType[iwpos])
					
					for temp in [sp1, sp2, sp3]:
						if len(temp) == 0:	# no shortest path, initial node and goal node are not connected in graph
							OUT.write('NA\n')
							continue				
						for i in range(len(temp)-1):
							if (temp[i], temp[i+1]) in g.edgeType:
								OUT.write(str(temp[i])+'|'+g.nodeType[temp[i]]+'\t'+g.edgeType[(temp[i],temp[i+1])]+'\t')
							else:
								print("edgeType error: ", ID, temp[i], temp[i+1])
						OUT.write(str(temp[len(temp)-1])+'|'+g.nodeType[temp[len(temp)-1]]+'\n')
						
					TripCnt = TripCnt + 1
					#for element in edgeSet:
					#	OUT.write(element)
					
		ID = arr[1]
		g = Graph()
		Count = Count + 1
	else:
		edgeDic[(int(arr[2]), int(arr[4]))] = line
		edgeDic[(int(arr[4]), int(arr[2]))] = line
		g.add_node(int(arr[2]), arr[1])
		g.add_node(int(arr[4]), arr[3])
		g.add_edge(int(arr[2]), int(arr[4]), 1, arr[0])


print("Total number of sentences read: ", Count)
print("Total number of triplets: ", TripCnt+1)
OUT.close()
