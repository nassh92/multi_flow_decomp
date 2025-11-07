import pickle
import networkx as nx
import time

"""graph_path_file = "data/original_graphs/lieusaint.gpickle"
with open(graph_path_file, 'rb') as f:
    g = pickle.load(f)

adj_mat = nx.to_numpy_array(g)

start_time = time.time_ns() / (10 ** 9)
for node in range(len(adj_mat)):
    successors = [succ for succ in range(len(adj_mat)) if adj_mat[node][succ]]
    predecessors = [pred for pred in range(len(adj_mat)) if adj_mat[pred][node]]
end_time = time.time_ns() / (10 ** 9)
print("The time adj_mat ", end_time - start_time)


start_time = time.time_ns() / (10 ** 9)
for node in g.nodes:
    successors = list(g.successors(node))
    predecessors = list(g.predecessors(node))
end_time = time.time_ns() / (10 ** 9)
print("The time networkx ", end_time - start_time)


start_time = time.time_ns() / (10 ** 9)
dfs_succs = nx.dfs_successors(g, source = list(g.nodes)[10])
end_time = time.time_ns() / (10 ** 9)
print("The time networkx ", end_time - start_time)
print("Nb successors ", len(dfs_succs))"""

G = nx.DiGraph()

G.add_edges_from([(0, 1), (1, 2), (2, 3)])

tab = [(n, nbrdict) for n, nbrdict in G.adjacency()]

print(tab)



