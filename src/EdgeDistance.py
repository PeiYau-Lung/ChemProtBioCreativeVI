
import numpy as np

'''
The Hierarchy of typed dependencies is defined by 
https://nlp.stanford.edu/software/dependencies_manual.pdf
and 
http://universaldependencies.org/u/dep/all.html
'''

class EdgeDistance(object):
	def __init__(self):
	# Define tree of dependency types

		self.HieDep = {}
		self.HieDep['root'] = 'root'
		self.HieDep['dep'] = 'root>dep'
		
		self.HieDep['acl'] = 'root>dep>acl'		# added by JZ, not sure whether this is the best place
		self.HieDep['acl:relcl'] = 'root>dep>acl:relcl'	# added by JZ, not sure whether this is the best place
		
		self.HieDep['aux'] = 'root>dep>aux'
		self.HieDep['auxpass'] = 'root>dep>aux>auxpass'
		self.HieDep['cop'] = 'root>dep>aux>cop'

		self.HieDep['arg'] = 'root>dep>arg'
		self.HieDep['agent'] = 'root>dep>arg>agent'

		self.HieDep['comp'] = 'root>dep>arg>comp'
		self.HieDep['acomp'] = 'root>dep>arg>comp>acomp'
		self.HieDep['ccomp'] = 'root>dep>arg>comp>ccomp'
		self.HieDep['xcomp'] = 'root>dep>arg>comp>xcomp'

		self.HieDep['obj'] = 'root>dep>arg>comp>obj'
		self.HieDep['dobj'] = 'root>dep>arg>comp>obj>dobj'
		self.HieDep['iobj'] = 'root>dep>arg>comp>obj>iobj'
		self.HieDep['pobj'] = 'root>dep>arg>comp>obj>pobj'

		self.HieDep['subj'] = 'root>dep>arg>subj'
		self.HieDep['nsubj'] = 'root>dep>arg>subj>nsubj'
		self.HieDep['nsubjpass'] = 'root>dep>arg>subj>nsubj>nsubjpass'	# added by JZ
		self.HieDep['csubj'] = 'root>dep>arg>subj>csubj'
		self.HieDep['csubjpass'] = 'root>dep>arg>subj>csubj>csubjpass'	# added by JZ
		
		self.HieDep['cc'] = 'root>dep>cc'
		self.HieDep['conj'] = 'root>dep>conj'
		self.HieDep['expl'] = 'root>dep>expl'
		self.HieDep['mod'] = 'root>dep>mod'
		self.HieDep['amod'] = 'root>dep>mod>amod'
		self.HieDep['appos'] = 'root>dep>mod>appos'
		self.HieDep['advcl'] = 'root>dep>mod>advcl'
		self.HieDep['det'] = 'root>dep>mod>det'
		self.HieDep['predet'] = 'root>dep>mod>predet'
		self.HieDep['preconj'] = 'root>dep>mod>preconj'
		self.HieDep['vmod'] = 'root>dep>mod>vmod'
		self.HieDep['mwe'] = 'root>dep>mod>mwe'
		self.HieDep['mark'] = 'root>dep>mod>mwe>mark'
		self.HieDep['advmod'] = 'root>dep>mod>advmod'
		self.HieDep['neg'] = 'root>dep>mod>advmod>neg'
		self.HieDep['rcmod'] = 'root>dep>mod>rcmod'
		self.HieDep['quantmod'] = 'root>dep>mod>quantmod'
		self.HieDep['nn'] = 'root>dep>mod>nn'
		self.HieDep['npadvmod'] = 'root>dep>mod>npadvmod'
		self.HieDep['npadvmod'] = 'root>dep>mod>npadvmod>tmod'
		self.HieDep['num'] = 'root>dep>mod>num'
		self.HieDep['number'] = 'root>dep>mod>number'
		self.HieDep['prep'] = 'root>dep>mod>prep'
		self.HieDep['poss'] = 'root>dep>mod>poss'
		self.HieDep['possesive'] = 'root>dep>mod>possessive'
		self.HieDep['prt'] = 'root>dep>mod>prt'
		self.HieDep['parataxis'] = 'root>dep>parataxis'
		self.HieDep['parataxis'] = 'root>dep>parataxis'
		self.HieDep['goeswith'] = 'root>dep>goeswith'
		self.HieDep['punct'] = 'root>dep>punct'
		self.HieDep['ref'] = 'root>dep>ref'
		self.HieDep['sdep'] = 'root>dep>sdep'
		self.HieDep['xsubj'] = 'root>dep>sdep>xsubj'
    
		# From Universal dependencies
		self.HieDep['discourse'] = 'root>dep>mod>discourse'
		self.HieDep['vocative'] = 'root>dep>vocative'
		self.HieDep['dislocated'] = 'root>dep>dislocated'
		self.HieDep['clf'] = 'root>dep>mod>clf'
		self.HieDep['case'] = 'root>dep>mod>case'
		self.HieDep['orphan'] = 'root>dep>orphan'
		self.HieDep['reparandum'] = 'root>dep>reparandum'
		self.HieDep['list'] = 'root>dep>list'
		self.HieDep['nmod'] = 'root>dep>mod>nmod'
		self.HieDep['nmod:npmod'] = 'root>dep>mod>nmod:npmod'	# added by JZ, may be this can be under nmod like root>dep>mod>nmod>nmod:npmod
		self.HieDep['nmod:poss'] = 'root>dep>mod>nmod:poss'		# added by JZ
		self.HieDep['nmod:tmod'] = 'root>dep>mod>nmod:tmod'     # added by PL on July 20
		self.HieDep['nummod'] = 'root>dep>mod>nummod'
		self.HieDep['flat'] = 'root>dep>flat'
		self.HieDep['fixed'] = 'root>dep>fixed'
		self.HieDep['compound'] = 'root>dep>compound'
		
	# we are calculating the distances between typed dependencies based on their distances on the hiereachical tree, which is
	# probably not optimal. This distance matrix can be optimized in the future.
	
    # the largest distance is set to 3, because the distance for deleting/inserting an edge/node is set to 1, if the distance is much greater 
	# than 2, the alignment can simply delete both edges (same as distance = 2). Here we use 3, which is kind of arbitrary. 
	# the purpose is that if two edges are very different, it is better not align them. With 3, we kind of say that they should not be aligned.
	def edge_dist(self, edge1, edge2):	   
		try:
			path1 = self.HieDep[edge1].split('>')
		except:
			print('Warning: ',edge1, ' is not in the Hierarchy of typed dependencies. The distance will set to 3.')
			return(3)
		try:
			path2 = self.HieDep[edge2].split('>')
		except:
			print('Warning: ',edge2, ' is not in the Hierarchy of typed dependencies. The distance will set to 3.')
			return(3)
		length1 = len(path1)
		length2 = len(path1)
		#print(self.HieDep[edge1])
		#print(self.HieDep[edge2])
		overlapping_node = np.intersect1d(path1,path2)
		if len(overlapping_node)==0:
			distance = length1-1 + length2-1
			return(distance*3/7) # we use 3/7 because the largest distance in the tree is 7
		else:
			if length1 <= length2:
				for i in np.arange(length1-1,-1,-1):
					if path1[i] in path2:
						idx_node1 = path1.index(path1[i])
						idx_node2 = path2.index(path1[i])
						#print path1[i],idx_node1,idx_node2
						distance = path1.index(edge1) - idx_node1 + path2.index(edge2) - idx_node2
						return(distance*3/7)
					else:
						continue
			else:                        
				for i in np.arange(length2-1,-1,-1):
					if path2[i] in path1:
						idx_node1 = path1.index(path2[i])
						idx_node2 = path2.index(path2[i])
						distance = path1.index(edge1) - idx_node1 + path2.index(edge2) - idx_node2
						return(distance*3/7)
					else:
						continue

    
