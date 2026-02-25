

"""import networkx as nx

G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edge(1, 2)

print(len(G))"""


import igraph as ig

G = ig.Graph()
G.add_vertices(3)
G.add_edges([(0, 1)])

print(len(G))