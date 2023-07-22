from typing import Tuple, List
from math import log, exp
import ccxt
import sys

# binance = ccxt.binance()

def get_currencies_rates_and_tickers():

    tickers = binance.fetch_tickers()

    currencies = []

    for key in tickers.keys():
        if tickers[key]['bid'] != None and tickers[key]['ask'] != None and '/' in key:
            currency_a, currency_b = key.split('/')
            currencies.append(currency_a)
            currencies.append(currency_b)

    currencies = list(set(currencies))

    rates = []

    for i in range(len(currencies)):
        rates.append([])
        for j in range(len(currencies)):
            if i == j:
                rates[i].append(1.0)
            else:
                rates[i].append(None)

    for key in tickers.keys():
        if tickers[key]['bid'] != None and tickers[key]['ask'] != None and '/' in key:
            currency_a, currency_b = key.split('/')
            rates[currencies.index(currency_a)][currencies.index(currency_b)] = tickers[key]['ask']
            rates[currencies.index(currency_b)][currencies.index(currency_a)] = 1.0 / tickers[key]['bid']

    return currencies, rates, tickers


rates = [
    [1, 0.23, 0.25, 16.43, 18.21, 4.94],
    [4.34, 1, 1.11, 71.40, 79.09, 21.44],
    [3.93, 0.90, 1, 64.52, 71.48, 19.37],
    [0.061, 0.014, 0.015, 1, 1.11, 0.30],
    [0.055, 0.013, 0.014, 0.90, 1, 0.27],
    [0.20, 0.047, 0.052, 3.33, 3.69, 1],
]

# rates = [
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1],
# ]

# print(rates)

# for i in range(len(rates)):
#     for j in range(i+1, len(rates[0])):
#         rates[i][j], rates[j][i] = rates[j][i], rates[i][j]
#         rates[i][j], rates[j][i] = 1, 1

# print(rates)

currencies = ('PLN', 'EUR', 'USD', 'RUB', 'INR', 'MXN')

edges = []
for i in range(len(rates)):
    for j in range(len(rates[0])):
        edge = (currencies[i], currencies[j], rates[i][j])
        edges.append(edge)

source = currencies[2]


# Python3 program for the above approach

# Structure to represent a weighted
# edge in graph
class Edge:
	def __init__(self):
		self.src = 0
		self.dest = 0
		self.weight = 0

# Structure to represent a directed
# and weighted graph
class Graph:

	def __init__(self):
		
		# V. Number of vertices, E.
		# Number of edges
		self.V = 0
		self.E = 0
		
		# Graph is represented as
		# an array of edges.
		self.edge = []
	
# Creates a new graph with V vertices
# and E edges
def createGraph(V, E):
	graph = Graph();
	graph.V = V;
	graph.E = E;
	graph.edge = [Edge() for i in range(graph.E)]
	return graph;

# Function runs Bellman-Ford algorithm
# and prints negative cycle(if present)
def NegCycleBellmanFord(graph, src):
    V = graph.V
    E = graph.E
    dist =[float('inf') for i in range(V)]
    parent =[-1 for i in range(V)]
    dist[src] = 0;

    # Relax all edges |V| - 1 times.
    for i in range(1, V):
        for j in range(E):

            u = graph.edge[j].src
            v = graph.edge[j].dest
            weight = graph.edge[j].weight

            if (dist[u] != float('inf') and
                dist[u] + weight < dist[v]):
            
                dist[v] = dist[u] + weight
                parent[v] = u

	# Check for negative-weight cycles
    C = -1
    for i in range(E):
        u = graph.edge[i].src
        v = graph.edge[i].dest
        weight = graph.edge[i].weight

        if (dist[u] != float('inf') and
            dist[u] + weight < dist[v]):
            
            # Store one of the vertex of
            # the negative weight cycle
            C = v
            break
		
    if (C != -1):	
        for i in range(V):	
            C = parent[C]

        # To store the cycle vertex
        cycle = []	
        v = C
        
        while (True):
            cycle.append(v)
            if (v == C and len(cycle) > 1):
                break
            v = parent[v]

        # Reverse cycle[]
        cycle.reverse()

        # Printing the negative cycle
        for v in cycle:	
            # print(v, end = " ")
            print(currencies[v], end=' ')
        print()
        m = 1.0
        w = cycle[0]
        for v in cycle[1:]:
            for e in graph.edge:
                if e.src == w and e.dest == v:
                    print(e.src, e.dest, exp(-e.weight))
                    m *= exp(-e.weight)
            w = v
        print('cumulative rate:', m)
        print()
    else:
        print(-1);

def f(edges, source):
    # initialize distances to infinity
    distances = {}
    for edge in edges:
        distances[edge[0]] = float('inf')
        distances[edge[1]] = float('inf')
    # set distance of source to zero
    distances[source] = 0
    # relax edges |V| - 1 times
    for i in range(len(distances)-1):
        for edge in edges:
            u, v, w = edge
            # if distance can be improved, update it
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
    # check for negative cycles and nodes to a set
    negative_cycle_nodes = set()
    for edge in edges:
        u, v, w = edge
        # if distance can still be improved, there is a negative cycle
        if distances[u] != float('inf') and distances[u] + w < distances[v]:
            negative_cycle_nodes.add(u)
            negative_cycle_nodes.add(v)
    # if no negative cycles, return distances
    print(negative_cycle_nodes)
    if not negative_cycle_nodes:
        return distances
    # otherwise, find all cycles that contain the nodes in the set
    visited = set()
    def dfs(node, path):
        # mark node as visited
        visited.add(node)
        # add node to path
        path.append(node)
        # check if path is a cycle
        if len(path) > 1 and path[0] == path[-1]:
            # print cycle
            print(path)
            # remove the last node from path and return
            path.pop()
            return
        # explore neighbours of node
        for edge in edges:
            if edge[0] == node and edge[1] in negative_cycle_nodes and not edge[1] in visited:
                dfs(edge[1], path)
        # remove node from path and unmark as visited
        path.pop()
        visited.remove(node)
    # start dfs from each node in the set
    for node in negative_cycle_nodes:
        dfs(node, [])
    # return none if there are negative cycles
    return None

def bellman_ford(edges, source):
  # Initialize distances to infinity
  distances = {}
  for edge in edges:
    distances[edge[0]] = float('inf')
    distances[edge[1]] = float('inf')
  # Set distance of source to zero
  distances[source] = 0
  # Relax edges |V| - 1 times
  for i in range(len(distances) - 1):
    for edge in edges:
      u, v, w = edge
      # If distance can be improved, update it
      if distances[u] + w < distances[v]:
        distances[v] = distances[u] + w
  # Check for negative cycles and add nodes to a set
  negative_cycle_nodes = set()
  for edge in edges:
    u, v, w = edge
    # If distance can still be improved, there is a negative cycle
    if distances[u] + w < distances[v]:
      negative_cycle_nodes.add(u)
      negative_cycle_nodes.add(v)
  # If no negative cycles, return distances
  if not negative_cycle_nodes:
    return distances
  # Otherwise, find all cycles that contain the nodes in the set
  visited = set()
  def dfs(node, path):
    # Mark node as visited
    visited.add(node)
    # Add node to path
    path.append(node)
    # Check if path is a cycle
    if len(path) > 1 and path[0] == path[-1]:
      # Print cycle
      print(path)
      # Remove last node from path and return
      path.pop()
      return
    # Explore neighbors of node
    for edge in edges:
      if edge[0] == node and edge[1] in negative_cycle_nodes and edge[1] not in visited:
        dfs(edge[1], path)
    # Remove node from path and unmark as visited
    path.pop()
    visited.remove(node)
  
  # Start dfs from each node in the set
  for node in negative_cycle_nodes:
    dfs(node, [])
  
  # Return None as there are negative cycles
  return None


# function BellmanFord(list vertices, list edges, vertex source) is

#     // This implementation takes in a graph, represented as
#     // lists of vertices (represented as integers [0..n-1]) and edges,
#     // and fills two arrays (distance and predecessor) holding
#     // the shortest path from the source to each vertex

#     distance := list of size n
#     predecessor := list of size n

#     // Step 1: initialize graph
#     for each vertex v in vertices do

#         distance[v] := inf             // Initialize the distance to all vertices to infinity
#         predecessor[v] := null         // And having a null predecessor
    
#     distance[source] := 0              // The distance from the source to itself is, of course, zero

#     // Step 2: relax edges repeatedly
    
#     repeat |V|−1 times:
#         for each edge (u, v) with weight w in edges do
#             if distance[u] + w < distance[v] then
#                 distance[v] := distance[u] + w
#                 predecessor[v] := u

#     // Step 3: check for negative-weight cycles
#     for each v in vertices do
#         u := predecessor[v]
#         if u is not null and distance[u] + weight of (u, v) < distance[v] then
#             // A negative cycle exist
#             visited := list of size n initialized with false
#             visited[v] := true
#             while not visited[u] do
#                 visited[u] := true
#                 u := predecessor[u]
#             // u is a vertex in a negative cycle, find the cycle itself
#             ncycle := [u]
#             v := predecessor[u]
#             while v != u do
#                 ncycle := concatenate([v], ncycle)
#                 v := predecessor[v]
#             error "Graph contains a negative-weight cycle", ncycle
#     return distance, predecessor


def BellmanFord(vertices, edges, source):

    # // This implementation takes in a graph, represented as
    # // lists of vertices (represented as integers [0..n-1]) and edges,
    # // and fills two arrays (distance and predecessor) holding
    # // the shortest path from the source to each vertex

    # distance := list of size n
    distance = len(vertices) * [float('inf')]
    # predecessor := list of size n
    predecessor = len(vertices) * [None]

    # // Step 1: initialize graph
    # for each vertex v in vertices do

        # distance[v] := inf             // Initialize the distance to all vertices to infinity
        # predecessor[v] := null         // And having a null predecessor
    
    # distance[source] := 0              // The distance from the source to itself is, of course, zero
    distance[source] = 0

    # // Step 2: relax edges repeatedly
    
    # repeat |V|−1 times:
    for i in range(len(vertices)-1):
        # for each edge (u, v) with weight w in edges do
        for edge in edges:
            u, v, w = edge
            # if distance[u] + w < distance[v] then
            if distance[u] + w < distance[v]:
                # distance[v] := distance[u] + w
                distance[v] = distance[u] + w
                # predecessor[v] := u
                predecessor[v] = u

    # // Step 3: check for negative-weight cycles
    # for each v in vertices do
    best_m = 1.0
    best_mncycle = []
    for v in vertices:
        # u := predecessor[v]
        u = predecessor[v]
        # if u is not null and distance[u] + weight of (u, v) < distance[v] then
        uv_edge = None
        for edge in edges:
            # print(edge)
            if edge[0] == u and edge[1] == v:
                uv_edge = edge
        if uv_edge == None:
            continue
        else:
            w = uv_edge[2]
        if u != None and distance[u] + w < distance[v]:
            # // A negative cycle exist
            # visited := list of size n initialized with false
            visited = len(vertices) * [False]
            # visited[v] := true
            visited[v] = True
            # while not visited[u] do
            while not visited[u]:
                # visited[u] := true
                visited[u] = True
                # u := predecessor[u]
                u = predecessor[u]
            # // u is a vertex in a negative cycle, find the cycle itself
            # ncycle := [u]
            ncycle = [u]
            # v := predecessor[u]
            v = predecessor[u]
            # while v != u do
            while v != u:
                # ncycle := concatenate([v], ncycle)
                ncycle = [v] + ncycle
                # v := predecessor[v]
                v = predecessor[v]
            print("Graph contains a negative-weight cycle", ncycle)
            mncycle = ncycle + [ncycle[0]]
            # mncycle = ncycle[::-1]
            # mncycle = mncycle + [mncycle[0]]
            m = 1.0
            for i in range(len(mncycle)-1):
                print(currencies[mncycle[i]], end=" ")
                for edge in edges:
                    if edge[0] == mncycle[i] and edge[1] == mncycle[i+1]:
                        print(exp(-edge[2]), end=" ")
                        m *= exp(-edge[2])
            print('cumulative rate:', m)
            if m > best_m and not currencies.index('GFT') in mncycle:
                best_m = m
                best_mncycle = mncycle
    return distance, predecessor, best_m, best_mncycle

# Driver Code
if __name__=='__main__':
	
    binance = ccxt.binance()
    tickers = binance.fetch_tickers()

    currencies = []
    for key in tickers.keys():
        if tickers[key]['bid'] != None and tickers[key]['ask'] != None and '/' in key:
            currency_a, currency_b = key.split('/')
            currencies.append(currency_a)
            currencies.append(currency_b)

    currencies = list(set(currencies))

    edge_count = 0
    for key in tickers.keys():
        if tickers[key]['bid'] != None and tickers[key]['ask'] != None and '/' in key:
            # currency_a, currency_b = key.split('/')
            # rates[currencies.index(currency_a)][currencies.index(currency_b)] = tickers[key]['ask']
            # rates[currencies.index(currency_b)][currencies.index(currency_a)] = 1.0 / tickers[key]['bid']
            edge_count += 2


    # Number of vertices in graph
    V = len(currencies);

    # Number of edges in graph
    E = edge_count;
    graph = createGraph(V, E);

    # k = 0
    # for i in range(len(rates)):
    #     for j in range(len(rates[0])):
    #         graph.edge[k].src = i
    #         graph.edge[k].dest = j
    #         graph.edge[k].weight = -log(rates[i][j])
    #         k += 1
    #         # edge = (currencies[i], currencies[j], rates[i][j])
    #         # edges.append(edge)

    k = 0
    for key in tickers.keys():
        if tickers[key]['bid'] != None and tickers[key]['ask'] != None and '/' in key:
            currency_a, currency_b = key.split('/')
            graph.edge[k].src = currencies.index(currency_b)
            graph.edge[k].dest = currencies.index(currency_a)
            graph.edge[k].weight = -log(tickers[key]['ask'])
            k += 1
            graph.edge[k].src = currencies.index(currency_a)
            graph.edge[k].dest = currencies.index(currency_b)
            graph.edge[k].weight = -log(1.0/tickers[key]['bid'])
            k += 1
    # print(k)

    k = 0
    edges = []
    for key in tickers.keys():
        if tickers[key]['bid'] != None and tickers[key]['ask'] != None and '/' in key:
            currency_a, currency_b = key.split('/')
            edge = (currencies.index(currency_a), currencies.index(currency_b), -log(1.0/tickers[key]['ask']))
            edges.append(edge)
            edge = (currencies.index(currency_b), currencies.index(currency_a), -log(tickers[key]['bid']))
            edges.append(edge)
    # print(len(edges))
    source = currencies.index('USDT')

    rates = []

    for i in range(len(currencies)):
        rates.append([])
        for j in range(len(currencies)):
            if i == j:
                rates[i].append(1.0)
            else:
                rates[i].append(None)

    for key in tickers.keys():
        if tickers[key]['bid'] != None and tickers[key]['ask'] != None and '/' in key:
            currency_a, currency_b = key.split('/') # ARB/USDT
            # assert(tickers[key]['ask'] > tickers[key]['bid'])
            # rates[currencies.index(currency_a)][currencies.index(currency_b)] = tickers[key]['ask']
            # rates[currencies.index(currency_b)][currencies.index(currency_a)] = 1.0 / tickers[key]['bid']
            rates[currencies.index(currency_a)][currencies.index(currency_b)] = 1.0/tickers[key]['ask'] # ARB/USDT
            rates[currencies.index(currency_b)][currencies.index(currency_a)] = tickers[key]['bid'] # USDT/ARB
            # rates[currencies.index(currency_a)][currencies.index(currency_b)] = tickers[key]['ask']
            # rates[currencies.index(currency_b)][currencies.index(currency_a)] = 1.0 / tickers[key]['bid']
	# Given Graph
	# graph.edge[0].src = 0;
	# graph.edge[0].dest = 1;
	# graph.edge[0].weight = 1;

	# graph.edge[1].src = 1;
	# graph.edge[1].dest = 2;
	# graph.edge[1].weight = 2;

	# graph.edge[2].src = 2;
	# graph.edge[2].dest = 3;
	# graph.edge[2].weight = 3;

	# graph.edge[3].src = 3;
	# graph.edge[3].dest = 4;
	# graph.edge[3].weight = -3;

	# graph.edge[4].src = 4;
	# graph.edge[4].dest = 1;
	# graph.edge[4].weight = -3;

	# Function Call
    # for source in range(V):
        # NegCycleBellmanFord(graph, source)
    # NegCycleBellmanFord(graph, currencies.index('USDT'))
    # bellman_ford(edges, source)
    # f(edges, source)
    vertices = range(len(currencies))
    # print('wikipedia')
    distance, predecessor, best_m, best_mncycle = BellmanFord(vertices, edges, source)
    print()
    print("Best negative-weight cycle", best_mncycle)
    # mncycle = ncycle + [ncycle[0]]
    mncycle = best_mncycle
    m = 1.0
    for i in range(len(mncycle)-1):
        print(currencies[mncycle[i]], end=" ")
        for edge in edges:
            if edge[0] == mncycle[i] and edge[1] == mncycle[i+1]:
                print(exp(-edge[2]), end=" ")
                m *= exp(-edge[2])
    print('cumulative rate:', m)
    balance = 10.0
    curr = 'USDT'
    end_curr = 'USDT'
    if m > 1.002:
        print('cumulative rate > 1.002, executing trades')
        print()
        print('IN:', balance, curr)
        print()

        # DRY RUN

        # buy in to the negative cycle with USDT
        print(curr, '-->', currencies[mncycle[0]])
        if currencies[mncycle[0]] + '/' + curr in tickers.keys():
            print('    buy', currencies[mncycle[0]], 'with', balance, curr, 'at rate', rates[mncycle[0]][currencies.index(curr)], end=' ')
            print(currencies[mncycle[0]] + '/' + curr, 'resulting in', balance * rates[mncycle[0]][currencies.index(curr)], end=' ')
            print(currencies[mncycle[0]])
            print('    (compared to', balance * 1.0/rates[currencies.index(curr)][mncycle[0]], ')')
            balance = balance * rates[mncycle[0]][currencies.index(curr)]
            curr = currencies[mncycle[0]]
        elif curr + '/' + currencies[mncycle[0]] in tickers.keys():
            print('    sell', balance, curr, 'for', currencies[mncycle[0]], 'at a rate of', rates[currencies.index(curr)][mncycle[0]], end=' ')
            print(currencies[mncycle[0]] + '/' + curr, 'resulting in', balance * rates[currencies.index(curr)][mncycle[0]], end=' ')
            print(currencies[mncycle[0]])
            print('    (compared to', balance * rates[mncycle[0]][currencies.index(curr)], ')')
            balance = balance * rates[currencies.index(curr)][mncycle[0]]
            curr = currencies[mncycle[0]]
        else:
            print('    ERROR: no ticker available')
            print('    printing all tickers for', currencies[mncycle[0]] + ':')
            for key in tickers.keys():
                if currencies[mncycle[0]] in key:
                    print(key)
            sys.exit()

        # execute the negative cycle 10 times
        for _ in range(2):
            for i in range(1, len(mncycle)):
                print(curr, '-->', currencies[mncycle[i]])
                if currencies[mncycle[i]] + '/' + curr in tickers.keys():
                    print('    buy', currencies[mncycle[i]], 'with', balance, curr, 'at rate', rates[mncycle[i]][currencies.index(curr)], end=' ')
                    print(currencies[mncycle[i]] + '/' + curr, 'resulting in', balance * rates[mncycle[i]][currencies.index(curr)], end=' ')
                    print(currencies[mncycle[i]])
                    print('    (compared to', balance * rates[mncycle[i]][currencies.index(curr)], ')')
                    balance = balance * 1.0/rates[currencies.index(curr)][mncycle[i]]
                    curr = currencies[mncycle[i]]
                elif curr + '/' + currencies[mncycle[i]] in tickers.keys():
                    print('    sell', balance, curr, 'for', currencies[mncycle[i]], 'at a rate of', rates[currencies.index(curr)][mncycle[i]], end=' ')
                    print(currencies[mncycle[i]] + '/' + curr, 'resulting in', balance * rates[currencies.index(curr)][mncycle[i]], end=' ')
                    print(currencies[mncycle[i]])
                    print('    (compared to', balance * rates[mncycle[i]][currencies.index(curr)], ')')
                    balance = balance * rates[currencies.index(curr)][mncycle[i]]
                    curr = currencies[mncycle[i]]
                else:
                    print('    ERROR: no ticker available')
                    print('    printing all tickers for', currencies[mncycle[i]] + ':')
                    for key in tickers.keys():
                        if currencies[mncycle[i]] in key:
                            print(key)
                    sys.exit()
        
        # buy out of negative cycle for USDT
        print(curr, '-->', end_curr)
        if end_curr + '/' + curr in tickers.keys():
            print('    buy', end_curr, 'with', balance, curr, 'at rate', rates[currencies.index(end_curr)][currencies.index(curr)], end=' ')
            print(end_curr + '/' + curr, 'resulting in', balance * rates[currencies.index(end_curr)][currencies.index(curr)], end=' ')
            print(end_curr)
            print('    (compared to', balance * rates[currencies.index(end_curr)][currencies.index(curr)], ')')
            balance = balance * 1.0/rates[currencies.index(curr)][currencies.index(end_curr)]
            curr = end_curr
        elif curr + '/' + end_curr in tickers.keys():
            print('    sell', balance, curr, 'for', end_curr, 'at a rate of', rates[currencies.index(curr)][currencies.index(end_curr)], end=' ')
            print(end_curr + '/' + curr, 'resulting in', balance * rates[currencies.index(curr)][currencies.index(end_curr)], end=' ')
            print(end_curr)
            print('    (compared to', rates[currencies.index(end_curr)][currencies.index(curr)], ')')
            balance = balance * rates[currencies.index(curr)][currencies.index(end_curr)]
            curr = end_curr
        else:
            print('    ERROR: no ticker available')
            print('    printing all tickers for', end_curr + ':')
            for key in tickers.keys():
                if currencies[mncycle[0]] in key:
                    print(key)
            sys.exit()

        print()
        print('OUT:', balance, curr)
        print()

                



