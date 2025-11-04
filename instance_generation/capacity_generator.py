import pprint

import random

import sys


def generate_capacities (adj_mat, base_capacity, factor, print_ = False):
    # Process center
    center = process_center(adj_mat, print_ = print_)
    if print_: print("Center ", center)

    # Initializations
    visited = [False]*len(adj_mat)
    pred = [None]*len(adj_mat)
    
    # Queue initialization
    visited[center] = True
    queue = [center]

    # Main loop
    while len(queue) > 0:
        # Dequeue a node
        node = queue.pop(0)
        # Browse through the successors of 'node' and if unvisited, update their distance from the source and their visited status
        for succ in range(len(adj_mat)):
            if adj_mat[node][succ] == 1 and not visited[succ]:
                visited[succ] = True
                pred[succ] = node 
                queue.append(succ)

    # Initialize the capacities
    caps = [[base_capacity*factor if pred[j]==i or pred[i]==j else base_capacity if adj_mat[i][j]==1 else 0 for j in range(len(adj_mat))]\
                                                                                                                for i in range(len(adj_mat))]
    if print_: print ("Predecessors ", pred)

    return caps, pred


def process_center(adj_mat, print_ = False):
    min_exentricity, centers = float("inf"), []
    # Main loop
    for node in range(len(adj_mat)):
        exentricity = process_exentricity (adj_mat, node)
        # Update the min
        if exentricity < min_exentricity: 
            centers = [node] 
            min_exentricity = exentricity
        elif exentricity == min_exentricity:
            centers.append(node)
    # Print
    if print_:
        print("All Centers ", centers)
        return centers[0]
    else:
        return centers[random.randint(0, len(centers)-1)]


def process_exentricity (adj_mat, source):
    # Initializations
    visited = [False]*len(adj_mat)
    distance = [0]*len(adj_mat)

    # Queue initialization
    visited[source] = True
    queue = [source]

    # Main loop
    while len(queue) > 0:
        # Dequeue a node
        node = queue.pop(0)
        # Browse through the successors of 'node' and if unvisited, update their distance from the source and their visited status
        for succ in range(len(adj_mat)):
            if adj_mat[node][succ] == 1 and not visited[succ]:
                visited[succ] = True
                distance[succ] = distance[node] + 1
                queue.append(succ)

    return distance[node]


if __name__ == "__main__":
    # Lien graphe : https://www.infoforall.fr/act/snt/graphes-et-liaisons-sociales/

    mat = [[0, 1, 1, 1, 0, 0, 0],
           [1, 0, 1, 0, 0, 1, 0],
           [1, 1, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 1, 0, 1, 1],
           [0, 1, 0, 1, 1, 0, 1],
           [0, 0, 0, 0, 1, 1, 0]]
    
    caps= generate_capacities (adj_mat = mat, 
                                base_capacity = 4, 
                                factor = 2, 
                                print_ = True)
    
    pprint.pprint(caps)
    

    
