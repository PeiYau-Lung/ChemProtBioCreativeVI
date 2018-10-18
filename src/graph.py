#!/usr/bin/python

class Graph(object):
    """
    A simple undirected, weighted graph
    """
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.distances = {}
        self.edgeType = {}
        self.nodeType = {}
		
    def add_node(self, value, type):
        self.nodes.add(value)
        self.nodeType[value] = type
		
    def add_edge(self, from_node, to_node, distance, type):
        self._add_edge(from_node, to_node, distance, type)
        self._add_edge(to_node, from_node, distance, type)

    def _add_edge(self, from_node, to_node, distance, type):
        self.edges.setdefault(from_node, [])
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = distance
        self.edgeType[(from_node, to_node)] = type

