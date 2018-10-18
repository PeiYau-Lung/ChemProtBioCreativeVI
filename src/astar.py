#!/usr/bin/python

import graph

def astar(graph, initial_node, goal_node, h):
    closed_set = set() # set of nodes already evaluated
    nodes = set() # set of tentative nodes to be evaluated
    nodes.add(initial_node)
 
    visited = {} # map of navigated nodes
    g_score = {initial_node: 0} # distance from start along optimal path
    h_score = {initial_node: h(initial_node, goal_node)} # heuristic estimate
    f_score = {initial_node: h_score[initial_node]} # estimated dist
    
    while nodes:
        
        x = None
        for node in nodes:
            if x is None:
                x = node
            elif f_score[node] < f_score[x]:
                x = node
        
        nodes.remove(x)
        if x == goal_node:
            return visited
 
        closed_set.add(x)
        for y in graph.edges[x]:
            if y in closed_set:
                continue
            tentative_g_score = g_score[x] + graph.distances[(x, y)]
 
            flag = False
            if y not in nodes or tentative_g_score < g_score[y]:
                nodes.add(y)
                flag = True
	    	 
            if flag:
                visited[y] = x
 
                g_score[y] = tentative_g_score
                h_score[y] = h(y, goal_node)
                f_score[y] = g_score[y] + h_score[y]
 
    return False
 
def shortest_path(graph, initial_node, goal_node, h):
    paths = astar(graph, initial_node, goal_node, h)
    route = [goal_node]
    #print paths
    #print route 
    if paths == False:
        return []

    while goal_node != initial_node:
        route.append(paths[goal_node])
        goal_node = paths[goal_node]
 
    route.reverse()
    return route
 
 
