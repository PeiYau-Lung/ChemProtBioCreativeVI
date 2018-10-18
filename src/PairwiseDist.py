'''
Usage:

Firstly, run CompIWPairDist.py to get pairwise distance matrix of interaction words (IW).
Next, from PairwiseDist import *
      PD = PairwiseDist()
      PD.IW_Dist(IW1,IW2)
      PD.IW_CosSim(IW1,IW2)
E.g. 
>>> PD.IW_Dist('effect','effects')
3.0155599999999998

'''


import numpy as np
from sklearn.cluster import KMeans

class PairwiseDist(object):
	
	def __init__(self):
		self.DistMatrix = np.loadtxt('./src/IW_DistMat.txt')
		self.CosSim = np.loadtxt('./src/IW_CosSim.txt')
		self.IWPosition = {}
		
		DIC = open('./src/IW_DistMat_dictionary.txt','r')
		for line in DIC:
			arr = line.strip().split('\t')
			#print(arr)
			self.IWPosition[arr[0]] = int(arr[1])
		
		self.IWCluster = {}
		kmean_label = KMeans(n_clusters=10, random_state=0).fit(self.DistMatrix).labels_
		#kmean_label = KMeans(n_clusters=10, random_state=0).fit(self.CosSim).labels_
		for i, IW in enumerate(self.IWPosition.keys()):
			self.IWCluster[IW] = kmean_label[self.IWPosition[IW]]

	def IW_Dist(self, IW1, IW2):
		try:
			pos1 = self.IWPosition[IW1]
		except:
			print('\''+IW1+'\' is not in the word vectors!\n')
					
		try:
			pos2 = self.IWPosition[IW2]
		except:
			print('\''+IW2+'\' is not in the word vectors!\n')

		return(self.DistMatrix[pos1, pos2])



	def IW_CosSim(self, IW1, IW2):
		try:
			pos1 = self.IWPosition[IW1]
		except:
			print('\''+IW1+'\' is not in the word vectors!\n')

		
		try:
			pos2 = self.IWPosition[IW2]
		except:
			print('\''+IW2+'\' is not in the word vectors!\n')
		
		return(self.CosSim[pos1,pos2])








