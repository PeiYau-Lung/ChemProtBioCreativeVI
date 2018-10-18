#!/usr/bin/python

# classes for shortest path and the shortest path sub-graph
import math
import sys
sys.path.append("./src/")
from EdgeDistance import *

class SPath(object):
	"""
	Class for shortest path
	"""
	def __init__(self, l):
		self.nodes = l
		self.edges = []
		self.nodeType = {}
		self.len = 0	# number of nodes
		
		if not len(l) == 0: 
			self.start = l[0]
			self.end = l[len(l)-1]
		else:
			self.start = -1
			self.end = -1

	def set_ends(self):
		if not len(self.nodes) == 0: 
			self.start = self.nodes[0]
			self.end = self.nodes[len(self.nodes)-1]
		else:
			self.start = -1
			self.end = -1

	def add_node(self, value, type):
		self.nodes.append(value)
		self.len = self.len + 1
		self.nodeType[value] = type
		
	def add_edge(self, value):
		self.edges.append(value)

	def path_copy(self, sp):
		sp.nodes = self.nodes
		sp.edges = self.edges
		sp.len = self.len
		sp.nodeType = self.nodeType.copy()
		
	# merger two shortest paths to form a new path
	def path_merge(self, sp):
		nsp = SPath()
		nsp.edges = self.edges + sp.edges
		nsp.nodes = self.nodes + sp.nodes[1:]
		nsp.len = self.len + sp.len - 1
		return nsp

	# reverse a shortest path, this will generate a copy of the current path
	def path_reverse(self):
		sp = Spath()
		sp.nodes = self.nodes
		sp.edges = self.edges
		sp.nodes.reverse()
		sp.edges.reverse()
		sp.len = self.len
		return sp
		
	# check whether two shortest paths are the same
	def path_equal(self, sp):
		if self.nodes == sp.nodes and self.edges == sp.edges:
			return True
		else:
			return False
	
	def path_dist(self, sp):
		"""
		This can actually be formulated as an alignment problem. 
		We simply align two shortest paths to compute the distance of their best alignment. 
		The penalities for gaps will be the edit distance. 
		We will only align edges since distances between nodes may only add to noise. 
		Maybe we can experiment with aligning nodes, as well, in the future.		
		There can be gaps in either shortest paths, and not every edge will be aligned in either shortest paths.
		We implement a dynamic programming algorithm to get the best alignment.
		https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm

		"""
		ED = EdgeDistance()
		DEL_DIST = 1	# distance for a single deletion of an edge
		#ssp = SPath([])	# shorter shortest path
		#lsp = SPath([]) # longer shortest path
		#if len(self) < len(sp):
		#	path_copy(self, ssp)
		#	path_copy(sp, lsp)
		#else:
		#	path_copy(self, lsp)
		#	path_copy(sp, ssp)
		
		#apath = []	# alignment path, which is a list of tuples, 
					# we ignore the first node and align the last node for convenience
		#SL = ssp.len -1 # number of nodes to be aligned for the shorter shortest path
		#LL = lsp.len -1	# number of nodes that can be aligned for the longer shortest path
		
		lenA, lenB = len(self.edges), len(sp.edges)
		
		score_matrix, trace_matrix = [], []
		for i in range(lenA + 1):
			score_matrix.append([None] * (lenB + 1))
			#if not score_only:
			trace_matrix.append([None] * (lenB + 1))
			
		for i in range(lenA + 1): 
			score_matrix[i][0] = i * DEL_DIST
		
		for i in range(lenB + 1):
			score_matrix[0][i] = i * DEL_DIST
		
		# Now, filling up the score and traceback matrices: 
		for row in range(1, lenA + 1):
			for col in range(1, lenB + 1):
				# Calculate the score that would occur by extending the 
				# alignment without gaps.
				#print(len(self.edges), len(sp.edges), row, col)
				nogap_score = score_matrix[row - 1][col - 1] + ED.edge_dist(self.edges[row - 1], sp.edges[col - 1]) 
				# Check the score that would occur if there were a gap in
				# self.edges. This could come from opening a new gap or
				# extending an existing one. 
				# A gap in self.edges can also be opened if it follows a gap in sp.edges
				
				col_score = score_matrix[row-1][col] + DEL_DIST
				row_score = score_matrix[row][col-1] + DEL_DIST
				best_score = min(nogap_score, col_score, row_score)
				score_matrix[row][col] = best_score
				
				if best_score == nogap_score:
					trace_matrix[row][col] = (row-1, col-1)
				elif best_score == col_score:
					trace_matrix[row][col] = (row-1, col)
				else:
					trace_matrix[row][col] = (row, col-1)
		
		return best_score 
								
class SPG(object):
	"""
	class for shortest path graph, which contains three shortest paths
	"""
	def __init__(self):
		self.SPs = []	# store three shortest paths, p1-iw, iw-p2, and p1-p2
		
		"""
		This is a list of lists, containing three shortest paths.
		The first is from p1 to iw (interaction word or relationship word);
		the second is from iw to p2;
		and the third is from p1 to p2.
		"""
		self.type = 0		
		self.rw = ""	# relationship word
		self.rw_type = -1	# the type of relationship word
		"""
		there are eight different shapes (or topology) or the graph formed by shortest paths
		here we make p1, the first protein/bio-entity name distinct from the second
		when computing distances, we will only match p1 of one graph with p1 of the other graph, and p2 with p2
		The eight types are numbered from 0 to 7, 8 for the type where at least one of the shortest path is empty (no path).
		"""
	def add_path(self, sp):
		self.SPs.append(sp)
	
	def clear_path(self):
		self.SPs = []
		
	def find_triplets(self):
		if len(self.SPs[0]) > 0:
			p1 = self.SPs[0][0]
		else:
			p1
		iw = self.SPs[1][0]
		p2 = self.SPs[2][len(self.SPs[2])-1]
		
	def computeType(self):
		"""
		The algorithm first finds the common node between pairs of shortest paths and among all three shortest paths.
		Using the common node information, it can assign the type of the shortest path graph.
		For example, the common node among all three shortest paths is iw for type 0, p1-iw-p2; the common node among all three shortest paths is p1 for type 1, iw-p1-p2, etc.
		"""
		if len(self.SPs[0]) == 0 or len(self.SPs[1]) == 0 or len(self.SPs[2]) == 0:
			self.type = 8
			return
		
		# common between pairs of shortest paths
		common_0_1 = list(set(self.SPs[0]).intersection(self.SPs[1])) # p1-iw and iw-p2
		common_0_2 = list(set(self.SPs[0]).intersection(self.SPs[2])) # p1-iw and p1-p2
		common_1_2 = list(set(self.SPs[1]).intersection(self.SPs[2])) # iw-p2 and p1-p2
		
		# common among all shortest paths
		common_all = list(set(common_0_1).intersection(self.SPs[2]))

		p1 = self.SPs[0][0]
		iw = self.SPs[1][0]
		p2 = self.SPs[2][len(self.SPs[2])-1]

		if len(common_all) == 1:
			# type 0, p1-iw-p2
			if common_all == [iw]:
				self.type = 0
			# type 1, iw-p1-p2
			elif common_all == [p1]:
				self.type = 1
			# type 2, p1-p2-iw
			elif common_all == [p2]:
				self.type = 2
			# type 3, the star shape with no other paths
			else: 
				self.type = 3
		
		# it is likely we will not have the following types because they form cycles
		# type 4, the triangle shape
		elif common_0_1 == [iw] and common_0_2 == [p1] and common_1_2 == [p2]:
			self.type = 4
		# type 5, the star shape with a path between p1 and p2
		elif len(common_0_1) > 1 and len(common_0_2) == 1 and len(common_1_2) == 1:
			self.type = 5
		# type 6, the star shape with a path between iw and p2
		elif len(common_0_2) > 1 and len(common_0_1) == 1 and len(common_1_2) == 1:
			self.type = 6
		# type 7, the star shape with a path between iw and p1
		elif len(common_1_2) > 1 and len(common_0_1) == 1 and len(common_0_2) == 1:
			self.type = 7
		else:
			print("A new shortest path graph type! ", self.SPs[0], self.SPs[1], self.SPs[2])
		
	def computeDist(self, spg):
		if (not self.type == spg.type) or self.type == 8 or spg.type == 8:
			print(self.type, spg.type)
			return 100000 # here we use 10000 as the largest distance between two SPGs
		
		# calculate relationship word distance using a simple approach for now
		if self.rw == spg.rw:
			rw_dist = 0
		else:
			rw_dist = 1
			
		if self.type == 0:
			dis1 = self.SPs[0].path_dist(spg.SPs[0])
			dis2 = self.SPs[1].path_dist(spg.SPs[1])
			return dis1 + dis2 + rw_dist
		elif self.type == 1:
			dis1 = self.SPs[0].path_dist(spg.SPs[0])
			dis2 = self.SPs[2].path_dist(spg.SPs[2])
			return dis1 + dis2 + rw_dist
		elif self.type == 2:
			dis1 = self.SPs[1].path_dist(spg.SPs[1])
			dis2 = self.SPs[2].path_dist(spg.SPs[2])
			return dis1 + dis2 + rw_dist
		elif self.type == 3:
			dis1 = self.SPs[0].path_dist(spg.SPs[0])
			dis2 = self.SPs[1].path_dist(spg.SPs[1])
			dis3 = self.SPs[2].path_dist(spg.SPs[2])
			return (dis1 + dis2	+ dis3) / 2 + rw_dist
		else:
			print("A new shortest path graph type! ", self.type)
			return 100000

		
