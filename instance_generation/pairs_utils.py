import sys

import os

import random

import itertools


sys.path.append(os.getcwd())
from instance_generation.maximum_flow_solver import EdmondKarpSolver
from utils.shortest_path_solvers import DijkstraShortestPathsSolver


PAIRS_GENERATION_TYPES = {"degree", "capacity", "min_cut", "all"}


###############################################################  Process the weights of the O/D pairs ################################################
def process_weight_pairs(pairs, graph_mat, pairs_generation = "degree"):
    """
    "graph_mat" can contain the adjacency matrix or the capacity matrix if 'pairs_generation == "degree"' or 'pairs_generation == "capacity"'.
    If 'pairs_generation == "min_cut"', graph_mat contains both the adjacency matrix and the capacities.
    """
    if  pairs_generation not in PAIRS_GENERATION_TYPES:
        print("Error, pairs_generation unrecognized. ", pairs_generation)
        sys.exit()
    
    if pairs_generation == "degree" or pairs_generation == "capacity":
        # All sources
        sources = [pair[0] for pair in pairs]
        # All destinations
        destinations = [pair[1] for pair in pairs]
        # A dict containing the sources associated with their weight (outdegree) 
        weight_sources = {node:sum(graph_mat[node][:]) for node in sources}
        # A dict containing the destinations associated with their weight (indegree)
        weight_destinations = {j:sum(graph_mat[i][j] for i in range(len(graph_mat))) for j in destinations}
        # Weights of the pairs
        weight_pairs = [weight_sources[pair[0]]+weight_destinations[pair[1]] for pair in pairs]

    elif pairs_generation == "min_cut":
        # Fetch the relevant graphs and process the weights of the pairs as their associated minimum cut value (max flow between the pairs)   
        adj_mat, capacities, transport_times, weight_pairs = graph_mat[0], graph_mat[1], graph_mat[2], []
        # Process the max flow between the source and destination of each pair in 'pairs'
        for (source, destination) in pairs:
            sp_solver = DijkstraShortestPathsSolver(source, adj_mat, transport_times, mode = "min_distance")
            sp_solver.run_dijkstra()
            sp_solver.construct_DAG_shortest_path (destination)
            dagsp_capacities = [[capacities[i][j] if sp_solver.dagsp[i][j] == 1 else 0 for j in range(len(capacities))] for i in range(len(capacities))]
            max_flow_solver = EdmondKarpSolver(sp_solver.dagsp, dagsp_capacities, source, destination)
            weight_pairs.append(max_flow_solver.run_edmond_karp())
    
    else:
        print("Weight pairs type not treated in this function.")
        sys.exit()
    
    return weight_pairs



##################  
#######                                Generate the O/D pairs using the degrees with adjacency matrix or the capacities with the matrix of capacities
##################
def generate_origin_destination_pairs_local (nb_pairs, graph_mat, nb_max_draws):
    """
    graph_mat can either be the adjacency matrix or the weights capacities 
    """
    # The weights associated to a node being a source calculated as the outdegree of the node
    weight_sources = [sum(graph_mat[i][:]) for i in range(len(graph_mat))]
    # The weights associated to a node being a destination calculated as the indegree of the node
    weight_destinations = [sum(graph_mat[i][j] for i in range(len(graph_mat))) for j in range(len(graph_mat))]
    # Choose randomly 'nb_pairs' of source-destination pairs according to 'weight_sources' and 'weight_destinations'
    pairs, num_draw = [], 0
    while len(pairs) < nb_pairs and num_draw < nb_max_draws:
        # Chose randomly a pair
        source = random.choices(list(range(len(graph_mat))), weights=weight_sources, k=1)[0]
        destination = random.choices(list(range(len(graph_mat))), weights=weight_destinations, k=1)[0]
        num_draw += 1
        # A new pair is drawn as long as the last drawn pair has already been chosen or the source is the same as the destination (with 
        # a maximum of 'nb_max_draws' draws)
        while ((source, destination) in pairs or source == destination) and num_draw < nb_max_draws:
            source = random.choices(list(range(len(graph_mat))), weights=weight_sources, k=1)[0]
            destination = random.choices(list(range(len(graph_mat))), weights=weight_destinations, k=1)[0]
            num_draw += 1
        # The last drawn pair is added if it has not already been generated (not in 'pairs') and the source is different from the destination
        if (source, destination) not in pairs and source != destination: pairs.append((source, destination))

    if len(pairs) == nb_pairs:
        return pairs
    else:
        print("Could not generate enough pairs.")
        sys.exit()



##################  
#######                        Generate the O/D pairs using the maximum flow           
##################
def generate_origin_destination_pairs_global (nb_pairs, adj_mat, capacities, nb_max_draws, transport_times):
    # For each possible O/D pair, process its weight as the value of the max flow from the source to the destination
    all_weights = []
    all_pairs = [(i, j) for i in range(len(adj_mat)) for j in range(len(adj_mat)) if i != j] 
    for (source, destination) in all_pairs:
        sp_solver = DijkstraShortestPathsSolver(source, adj_mat, transport_times, mode = "min_distance")
        sp_solver.run_dijkstra()
        sp_solver.construct_DAG_shortest_path (destination)
        dagsp_capacities = [[capacities[i][j] if sp_solver.dagsp[i][j] == 1 else 0 for j in range(len(capacities))] for i in range(len(capacities))]
        max_flow_solver = EdmondKarpSolver(sp_solver.dagsp, dagsp_capacities, source, destination)
        all_weights.append(max_flow_solver.run_edmond_karp())
    
    # Generate 'nb_pairs' pairs using the weights calculated previously
    pairs, nb_draws = [], 0
    while len(pairs) < nb_pairs and nb_draws < nb_max_draws:
        # Chose randomly a pair
        pair = random.choices(all_pairs, weights=all_weights, k=1)[0]
        nb_draws += 1
        # A new pair is drawn as long as the last drawn pair has already been chosen or the source is the same as the destination (with 
        # a maximum of 'nb_max_draws' draws)
        while pair in pairs and nb_draws < nb_max_draws:
            pair = random.choices(all_pairs, weights=all_weights, k=1)[0]
            nb_draws += 1
        # The last drawn pair is added if it has not already been generated (not in 'pairs')
        if pair not in pairs: pairs.append(pair)
    
    if len(pairs) == nb_pairs:
        return pairs
    else:
        print("Could not generate enough pairs.")
        sys.exit()