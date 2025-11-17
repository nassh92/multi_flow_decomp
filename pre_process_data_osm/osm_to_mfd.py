import networkx as nx
import pickle
import sys
import os
import urllib.parse
import random
import json
import numpy as np
import osmnx as ox
import math
from copy import deepcopy

import shapely
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge, substring
from math import radians, cos, sin, asin, sqrt

sys.path.append(os.getcwd())
from instance_generation.generic_multi_flow_instances_generator import generate_multi_flow_instance, fetch_ajust_ncorrect_flow_network_data
from msmd.multi_flow_desag_instance_utils import MultiFlowDesagInstance
from instance_generation.pairs_utils import process_weight_pairs, generate_origin_destination_pairs_local
from pre_process_data_osm.restrict_segments import show_duplicate_arcs_subgraph
from utils.graph_utils import construct_predecessors_list, init_graph_arc_attribute_vals, get_arcs, delete_arc


###############################################################################################
##
#################################### Convert networkx to matrices #############################
##
###############################################################################################

def return_capacities_data(nx_graph, graph, node_list, edge_list, car_size):
    min_capacity, max_capacity = float("inf"), 0
    capacities_data = init_graph_arc_attribute_vals(graph)
    for i, j in edge_list:
        u, v = node_list[i], node_list[j]
        capacities_data[i][j] = math.ceil(nx_graph[u][v]["lanes"] * nx_graph[u][v]["length"] / car_size)
        #capacities_data[i][j] = 100
        if capacities_data[i][j] > 0 and min_capacity > capacities_data[i][j]:
            min_capacity = capacities_data[i][j]
        if capacities_data[i][j] > max_capacity:
            max_capacity = capacities_data[i][j]
    print("Min capacity ", min_capacity)
    print("Max capacity ", max_capacity)
    return capacities_data


def return_transport_times_data(nx_graph, graph, node_list, edge_list):
    transport_times_data = init_graph_arc_attribute_vals(graph, init_val = float("inf"))
    for i, j in edge_list:
        u, v = node_list[i], node_list[j]
        transport_times_data[i][j] = nx_graph[u][v]["length"] / nx_graph[u][v]["maxspeed"]
    return transport_times_data


"""def return_network_matrices(g, car_size = 1, print_ = False):
    # Adjacency matric
    adj_mat = nx.to_numpy_array(g)
    if print_: print("Matrix size ", len(adj_mat))
    node_list = list(g.nodes)
    if print_: print("Node list size ", len(node_list))
    edge_list = list(g.edges)
    if print_: print("Edge list size ", len(edge_list))
    # Capacity matrix : capacity = (number of lanes * length) / typical size of a car
    capacity_mat = return_capacity_matrix(g, node_list, car_size)
    # Ideal times matrix = length / maxspeed
    transport_times_mat = return_transport_times_matrix(g, node_list)
    return adj_mat, capacity_mat, transport_times_mat, node_list, edge_list"""


def return_network_data(nx_graph, car_size = 1, print_ = False, matrix_representation = True):
    node_list = list(nx_graph.nodes)
    edge_list = [(node_list.index(u), node_list.index(v)) for u, v in nx_graph.edges]
    # Adjacency matrice
    if matrix_representation:
        graph = nx.to_numpy_array(nx_graph).tolist()
    else:
        graph = {node_list.index(node):[node_list.index(succ_node) for succ_node in successors_dict.keys()] 
                                                                        for node, successors_dict in nx_graph.adjacency()}
    # Capacity matrix : capacity = (number of lanes * length) / typical size of a car
    capacities_data = return_capacities_data(nx_graph, graph, node_list, edge_list, car_size)
    # Ideal times matrix = length / maxspeed
    transport_times_data = return_transport_times_data(nx_graph, graph, node_list, edge_list)

    if print_: 
        print("Network size ", len(graph))
        print("Node list size ", len(node_list))
        print("Edge list size ", len(edge_list))
        nb_sim = 0
        for u, v in edge_list:
            if (v, u) in edge_list: nb_sim += 1
        print("Number of symetric edges ", nb_sim/2)
    return graph, capacities_data, transport_times_data, node_list, edge_list




###############################################################################################
##
#################################### Attribute level pre-process ####################################
##
###############################################################################################

def aggregate_attributes(g):
    """ 
    Aggregate the attributes "maxspeed", "length" and "lanes" if they are of type 'list'
    "length" : The new length of a segment is the sum of the length of the segments composing it
    "lanes" : The number of lanes of the segment is the average #lanes of all the segments composing it weighted by their "length"
    "maxspeed" : The new maxspeed is the average maxspeed of a vehicule traversing the segment (sum of lengths/sum of traversal times)   
    """
    for u, v, k in g.edges(keys = True):
        # In the case where the attributes "maxspeed", "length" and "lanes" are all
        # lists of the same length, it is possible to process them and keep consistency
        if isinstance(g[u][v][k]["maxspeed"], list) and\
           isinstance(g[u][v][k]["length"], list) and\
            len(g[u][v][k]["maxspeed"]) == len(g[u][v][k]["length"]):
            sum_lengths, nb_seg = sum(g[u][v][k]["length"]), len(g[u][v][k]["length"])

            g[u][v][k]["maxspeed"] = sum_lengths/sum(g[u][v][k]["length"][m]/g[u][v][k]["maxspeed"][m] for m in range(nb_seg))
        
            g[u][v][k]["length"] = sum_lengths

            if isinstance(g[u][v][k]["lanes"], list):
                g[u][v][k]["lanes"] = min(g[u][v][k]["lanes"][m] for m in range(nb_seg))
        
        else:
            # Treat each attribute separatly if no consistency can be obtained 
            if isinstance(g[u][v][k]["maxspeed"], list):
                g[u][v][k]["maxspeed"] = min(g[u][v][k]["maxspeed"])
            
            if isinstance(g[u][v][k]["length"], list):
                g[u][v][k]["length"] = sum(g[u][v][k]["length"])
            
            if isinstance(g[u][v][k]["lanes"], list):
                g[u][v][k]["lanes"] = min(g[u][v][k]["lanes"])




###############################################################################################
##
#################################### Graph level pre-process ####################################
##
###############################################################################################

def multidigraph_to_digraph (g):
    """
    Transform a nx.MultiDigraph instance representing a city to a nx.DiGraph instance
    by removing the loops and the multiple edges from it.
    
    Treat each edge in the following way :
    1. If it is a loop, delete it.
    2. Else, if it has an associated parallel edge
        1.1 If there is only one edge among them with two ways (attribute oneway == False) keep that edge.
        1.2 Else, keep the edge with the greater number of lanes (attribute lanes of the edge).
        1.3 If multiple parallel edges have the same number of lanes, keep the one with highest 'max_speed' attribute. 
        1.4 If the are still multiple parallel edges keep the geometrically shortest one.
        1.4 In last resort, keep one of the remaining parallel edges randomly.

    Modifies g inplace. 
    """
    for u, v in set(g.edges()):
        if u == v: # Remove the self loop
            for uu, vv, kk in set(g.edges(keys=True)): 
                if uu == u and vv == v: g.remove_edge(uu, vv, kk)
        
        elif len(g[u][v]) > 1: # Check if (u, v) is a multiedge
            # Filter with multiway criteria
            parallel_uvk = {edge for edge in g.edges(keys=True) if edge[0] == u and edge[1] == v}
            multiway_edges_uv = {(uu, vv, kk) for uu, vv, kk, oneway in g.edges(data="oneway", keys=True, default=True) 
                                                if (uu, vv, kk) in parallel_uvk and not oneway}
            if len(multiway_edges_uv) == 1: # If there is exactly one edge, remove all other parallel edges
                to_be_removed = parallel_uvk - multiway_edges_uv
                for uu, vv, kk in to_be_removed: g.remove_edge(uu, vv, kk)
                continue
            # Keep the two way roads if any 
            remaining_parallel_uvk = parallel_uvk if len(multiway_edges_uv) == 0 else multiway_edges_uv
            
            # Filter with number of lanes criteria
            # Type of "lanes" instances of 'str' and 'int''
            max_lanes = max(g[uu][vv][kk]["lanes"] for uu, vv, kk in remaining_parallel_uvk)
            max_lanes_edges_uv = {(uu, vv, kk) for uu, vv, kk in remaining_parallel_uvk 
                                                    if g[uu][vv][kk]["lanes"] == max_lanes}
            if len(max_lanes_edges_uv) == 1: # If there is exactly one edge, remove all other parallel edges
                to_be_removed = remaining_parallel_uvk - max_lanes_edges_uv
                for uu, vv, kk in to_be_removed: g.remove_edge(uu, vv, kk)
                continue
            remaining_parallel_uvk = max_lanes_edges_uv

            # Filter with max speed criteria
            max_speed = max(g[uu][vv][kk]["maxspeed"] for uu, vv, kk in remaining_parallel_uvk)
            max_speed_edges_uv = {(uu, vv, kk) for uu, vv, kk in remaining_parallel_uvk 
                                                    if g[uu][vv][kk]["maxspeed"] == max_speed}
            if len(max_speed_edges_uv) == 1: # If there is exactly one edge, remove all other parallel edges
                to_be_removed = remaining_parallel_uvk - max_speed_edges_uv
                for uu, vv, kk in to_be_removed: g.remove_edge(uu, vv, kk)
                continue
            remaining_parallel_uvk = max_speed_edges_uv if len(max_speed_edges_uv) > 0 else remaining_parallel_uvk

            # Filter with max speed shortest path criteria
            val_shortest_edge = min(g[uu][vv][kk].get("length", float("inf")) for uu, vv, kk in remaining_parallel_uvk)
            shortest_edges_uv = [(uu, vv, kk) for uu, vv, kk in remaining_parallel_uvk if g[uu][vv][kk]["length"] == val_shortest_edge]
            # Random choice
            ix_edge_to_keep = np.random.choice(range(len(shortest_edges_uv)))
            to_be_removed = remaining_parallel_uvk - {shortest_edges_uv[ix_edge_to_keep]}
            for uu, vv, kk in to_be_removed: g.remove_edge(uu, vv, kk)
    
    return nx.DiGraph(g)


def delete_non_reach_PI_nodes(g, interest_point_ids):
    """
    Take a nx.DiGraph as input and keep all the nodes which are accessible from the interest points
    and from which the interest points can be reached 
    Modifies g inplace.
    """
    to_be_removed = set()
    eligible_nodes = {node for node in list(g.nodes) if node not in interest_point_ids}
    for node in eligible_nodes:
        # Get the descendants of the current node
        descendants = nx.descendants(g, node)
        # Mark the current node as to be removed if it has no descendants among the points of interest
        if len(descendants & interest_point_ids) == 0:
            to_be_removed.add(node)
            continue
        
        # Get the ancestors of the current node
        ancestors = nx.ancestors(g, node)
        # Mark the current node as to be removed if it has no ancesstors among the points of interest
        if len(ancestors & interest_point_ids) == 0:
            to_be_removed.add(node)
    
    # Remove the nodes
    for node in to_be_removed: g.remove_node(node)
    return len(to_be_removed)


def delete_non_reach_pairs_nodes(g, list_id_pairs, interest_point_ids):
    """
    Take a nx.DiGraph as input and keep all the nodes which are accessible from the interest points 
    Modifies g inplace.
    """
    to_be_removed = set()
    eligible_nodes = {node for node in list(g.nodes) if node not in interest_point_ids}
    for node in eligible_nodes:
        found = False
        for pair in list_id_pairs:
            # Get the source and the destination
            source_id, destination_id = pair
            
            # Get the descendants of the source and test if the current node belong to them
            descendants = nx.descendants(g, source_id)
            if node not in descendants: continue
            
            # Get the ancesstors of the destination and test if the current node belong to them
            ancestors = nx.ancestors(g, destination_id)
            if node in ancestors:
                found = True
                break
        if not found: to_be_removed.add(node)  
        
    # Remove the nodes
    for node in to_be_removed: g.remove_node(node)


def delete_waste_nodes(g, tabu_nodes = set()):
    """
    Take a nx.DiGraph as input and remove all 'wastes' nodes from it, which
    are the intermediary nodes and the deadends
    Modifies g in place
    """
    nb_removed_nodes = 0
    ls_nodes = [node for node in list(g.nodes) if node not in tabu_nodes]
    for node in ls_nodes:
        # Get predecessors
        predecessors = list(g.predecessors(node))
        in_degree = len(predecessors)
        # Get predecessors
        successors = list(g.successors(node))
        out_degree = len(successors)
        # Get neighbors 
        neighbors = set(predecessors+successors)
        # Delete the 'waste' nodes
        if len(neighbors) == 1: # Deadend
            g.remove_node(node)
            nb_removed_nodes += 1
        # Intermediary nodes
        elif (len(neighbors) == 2) and\
            ((out_degree == 2 and in_degree == 2) or (out_degree == 1 and in_degree == 1)):
            for pred in predecessors:
                for succ in successors:
                    if pred != succ:
                        # Process the new length attribute
                        l_pred = g[pred][node]["length"]
                        l_succ = g[node][succ]["length"]
                        new_l = l_pred + l_succ
                        # Process the new lanes attribute
                        lanes_pred = g[pred][node]["lanes"]
                        lanes_succ = g[node][succ]["lanes"]
                        new_lanes = (lanes_pred*l_pred + lanes_succ*l_succ) / new_l
                        # Process the new max_length attribute
                        max_speed_pred = g[pred][node]["maxspeed"]
                        max_speed_succ = g[node][succ]["maxspeed"]
                        new_max_speed = (l_pred + l_succ)/(l_pred/max_speed_pred + l_succ/max_speed_succ)
                        # Process the new geometry attribute
                        geom_pred = g[pred][node].get("geometry", None)
                        geom_succ = g[node][succ].get("geometry", None)
                        # Process new geometry depending on values of 'geom_pred' and 'geom_succ'
                        if geom_pred is None and geom_succ is None:
                           new_geom = None 
                        if geom_pred is None and geom_succ is not None:
                            new_geom = geom_succ
                        elif geom_pred is not None and geom_succ is None:
                            new_geom = geom_pred
                        elif geom_pred is not None and geom_succ is not None:
                            new_geom = shapely.union(geom_pred, geom_succ)
                        # Calculate optional parameters
                        if new_geom is None:
                            params = {"length":new_l, "lanes":new_lanes, "maxspeed":new_max_speed}
                        else:
                            params = {"length":new_l, "lanes":new_lanes, "maxspeed":new_max_speed, "geometry":new_geom}
                        # Add the new edge
                        g.add_edge(pred, 
                                   succ, 
                                   **params)
            
            # Remove node
            g.remove_node(node)
            nb_removed_nodes += 1
    return nb_removed_nodes




###############################################################################################
##
#################################### Interest Points stuff ####################################
##
###############################################################################################

def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance between two (lat, lon) pairs"""
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371000  # Radius of Earth in meters
    return c * r


def closest_nodes(g, P, k=1):
    """
    Find k closest nodes in DiGraph g to point P (lat, lon).
    Returns list of node IDs.
    """
    lat, lon = P
    distances = []
    
    for node, data in g.nodes(data=True):
        dist = haversine(lat, lon, data['y'], data['x'])
        distances.append((node, dist))
    
    # sort by distance
    distances.sort(key=lambda x: x[1])
    
    return [node for node, _ in distances[:k]]


def load_interest_points (file_path):
    # Open and load the interests points
    with open(file_path, "r", encoding="utf-8") as file:
        interest_points = json.load(file)
    return interest_points


def return_augmented_interest_points(g, list_interest_points):
    """
    Return a list of dict associating each point of interst to node in the graphe g
    """
    augmented_ls_point_of_interests = []
    for point_of_interest in list_interest_points:
        n_id = closest_nodes(g, (point_of_interest["lat"], point_of_interest["lon"]))[0]
        augmented_ls_point_of_interests.append(
            {
                "name":point_of_interest["nom"],
                "lat":point_of_interest["lat"],
                "lon":point_of_interest["lon"],
                "id":n_id
            }
        )
    return augmented_ls_point_of_interests


def generate_origin_destination_pairs (g, nb_pairs, ls_IPs, car_size = 1, nb_max_draws = 10):
    # get the node list
    IP_node_list = [interest_point["id"] for interest_point in ls_IPs]
    # The weights associated to a node being a source calculated as the outdegree capacity of the node
    weight_sources = [sum(math.ceil(g[u][v]["lanes"] * g[u][v]["length"] / car_size) for v in list(g.nodes) if g.has_edge(u, v)) for u in IP_node_list]
    # The weights associated to a node being a destination calculated as the indegree capacity of the node
    weight_destinations = [sum(math.ceil(g[u][v]["lanes"] * g[u][v]["length"]) / car_size for u in list(g.nodes) if g.has_edge(u, v)) for v in IP_node_list]
    # Choose randomly 'nb_pairs' of source-destination pairs according to 'weight_sources' and 'weight_destinations'
    pairs, num_draw = [], 0
    while len(pairs) < nb_pairs and num_draw < nb_max_draws:
        # Chose randomly a pair
        id_source = random.choices(IP_node_list, weights=weight_sources, k=1)[0]
        id_destination = random.choices(IP_node_list, weights=weight_destinations, k=1)[0]
        num_draw += 1
        # A new pair is drawn as long as the last drawn pair has already been chosen 
        # or the source is the same as the destination (with a maximum of 'nb_max_draws' draws)
        while ((id_source, id_destination) in pairs or id_source == id_destination) and num_draw < nb_max_draws:
            id_source = random.choices(IP_node_list, weights=weight_sources, k=1)[0]
            id_destination = random.choices(IP_node_list, weights=weight_destinations, k=1)[0]
            num_draw += 1
        # The last drawn pair is added if it has not already been generated (not in 'pairs') and the source is different from the destination
        if (id_source, id_destination) not in pairs and id_source != id_destination: pairs.append((id_source, id_destination))

    if len(pairs) == nb_pairs:
        return pairs
    else:
        print("Could not generate enough pairs.")
        sys.exit()




###########################################################################################
#
##############################  Converting osm data do Multi Flow Desagg instance
#
###########################################################################################

def pre_process_networkx(graph_path_file, 
                         interset_points_file_path,
                         nb_pairs,
                         nb_max_draws_pairs = 10,
                         car_size = 1,
                         save_dir = None,
                         generate_figure = None,
                         matrix_representation = True,
                         print_ = True):
    """
    Construct the instance

    """
    # Open the graph path file
    with open(graph_path_file, 'rb') as f:
        nx_graph = pickle.load(f)

    # Affichage
    if generate_figure is not None and generate_figure[0]:
        print("Figure generation - network before preprocessing.")
        fig, ax = ox.plot_graph(
                                nx.MultiDiGraph(nx_graph),
                                figsize=(60, 60),
                                node_color="yellow",
                                node_size=10,
                                edge_linewidth=2,
                                edge_alpha=0.4,   # transparency (0 = fully transparent, 1 = opaque)
                                node_alpha=0.4,
                                bgcolor="white",
                                close=True,
                                show = False,
                                save = save_dir is not None,
                                filepath = save_dir+"/before_preprocessing.png"
                            )
        fig.suptitle("Network before preprocessing.")
    
    # Merge the attributes which are in the form of lists
    aggregate_attributes(nx_graph)

    # Transform the multigraph to a graph instance by deleting the loop and filtering the parallel edges
    nx_graph = multidigraph_to_digraph (nx_graph)

    print("---------------Network size after aggregate attributes and MultiDiGraph to Digraph--------")
    print("Number of nodes ", len(nx_graph.nodes))
    print("Number of edges ", len(nx_graph.edges))
    print("------------------------------------------------------------------------------------------")

    # Load the interest points
    list_interest_points = load_interest_points (interset_points_file_path)
    list_interest_points = return_augmented_interest_points(nx_graph, list_interest_points)
    
    # Construct the pairs and their associated weights
    list_id_pairs = generate_origin_destination_pairs (nx_graph,
                                                       nb_pairs, 
                                                       list_interest_points, 
                                                       car_size,
                                                       nb_max_draws = nb_max_draws_pairs)
    
    # Delete all the nodes which can't be reached from the source and from which the destination can't be reached
    interest_point_ids = {node["id"] for node in list_interest_points}
    delete_non_reach_pairs_nodes(nx_graph, list_id_pairs, interest_point_ids)
    
    print("---------------Network size after deleting non reachable from pairs ------------------")
    print("Number of nodes ", len(nx_graph.nodes))
    print("Number of edges ", len(nx_graph.edges))
    print("--------------------------------------------------------------------------------------")
    
    # To be done : Delete the roundabouts 
    # Method : use a datastructure which ranks the nodes by closeness

    nb_removed_nodes = float('inf')
    while nb_removed_nodes > 0:
        # Delete the waste nodes (intermediary 2 neighbours, and one neighbour)
        nb_removed_nodes = delete_waste_nodes(nx_graph, interest_point_ids)
        print("Nb removed nodes ", nb_removed_nodes)

    print("------------------------- Network size after deleting waste nodes -------------------")
    print("Number of nodes ", len(nx_graph.nodes))
    print("Number of edges ", len(nx_graph.edges))
    print("--------------------------------------------------------------------------------------")


    print("----------------------------- Constructin the adjacency matrix ----------------------------------------")
    graph, capacities, transport_times, node_list, edge_list = return_network_data(nx_graph,
                                                                                   car_size, 
                                                                                   print_ = print_,
                                                                                   matrix_representation = matrix_representation)
    print("-------------------------------------------------------------------------------------------------------")

    # Retrieve the pairs after node deletion
    node_list = list(nx_graph.nodes)
    pairs = [(node_list.index(id_source), 
              node_list.index(id_destination)) for id_source, id_destination in list_id_pairs]

    # Process the weights associated to the pairs
    weight_pairs = process_weight_pairs(pairs, 
                                        graph, 
                                        arc_attribute_vals = capacities,
                                        predecessors_list = construct_predecessors_list(graph),
                                        pairs_generation = "capacity")

    # Affichage
    if generate_figure is not None and generate_figure[1]:
        print("Figure generation - network after preprocessing.")
        fig, ax = ox.plot_graph(
                                nx.MultiDiGraph(nx_graph),
                                figsize=(60, 60),
                                node_color="yellow",
                                node_size=10,
                                edge_linewidth=2,
                                edge_alpha=0.4,   # transparency (0 = fully transparent, 1 = opaque)
                                node_alpha=0.4,
                                bgcolor="white",
                                close=True,
                                show = False,
                                save = save_dir is not None,
                                filepath = save_dir+"/after_preprocessing.png"
                            )
        fig.suptitle("Network after preprocessing.")

    # Construct and return the dict containing all the informations related to the generated instance (follwing the value of 'pairs_generation') 
    return_dict = {"graph":graph,
                   "arcs":edge_list,
                   "nodes":node_list,
                   "capacities":capacities,
                   "transport_times":transport_times,
                   "pairs":pairs,
                   "weight_pairs":weight_pairs}
    return return_dict, nx_graph, list_interest_points



def construct_real_instances (graph_nx_path_file, 
                              interest_points_file_path,
                              dir_save_name_graph,
                              dir_save_name_multiflow,
                              dir_save_name_mfd,
                              nb_instances,
                              nb_pairs,
                              suffix_fname,
                              car_size = 1,
                              min_fl = 1,
                              nb_max_draws_pairs = 10,
                              nb_it_print = None,
                              save_dir = None,
                              matrix_representation = True,
                              generate_figure = None):
    # The file names associated to the the folder 'MI' containing the structural data (the arc, the capacities and the transport times)
    return_dict, _, _ = pre_process_networkx(graph_nx_path_file, 
                                             interest_points_file_path,
                                             nb_pairs,
                                             nb_max_draws_pairs = nb_max_draws_pairs,
                                             car_size = car_size,
                                             save_dir = save_dir,
                                             matrix_representation = matrix_representation,
                                             generate_figure = generate_figure)
    
    graph = return_dict["graph"]
    capacities = return_dict["capacities"]
    raw_transport_times = return_dict["transport_times"]
    all_pairs = return_dict["pairs"]
    weight_pairs = return_dict["weight_pairs"]
    if "desired_flow_values" in return_dict:
        all_desired_flow_values = [int(float(str_fl)) for str_fl in return_dict["desired_flow_values"]]
    else:
        all_desired_flow_values = [float("inf") for _ in range(nb_pairs)]
    
    np.save(os.path.join(dir_save_name_graph, 
                         "real_instance_"+suffix_fname), 
                         return_dict)
    
    # The loop
    dict_instances = {}
    for num_instance in range(nb_instances):
        # Some printing
        print("Treating instances numero ", num_instance)
        # Generation of the multiflow
        return_multi_flow_dict = generate_multi_flow_instance(deepcopy(graph), 
                                                              deepcopy(capacities), 
                                                              deepcopy(raw_transport_times), 
                                                              deepcopy(all_pairs), 
                                                              deepcopy(all_desired_flow_values), 
                                                              min_fl = min_fl, 
                                                              nb_it_print = nb_it_print,
                                                              weight_pairs = deepcopy(weight_pairs),
                                                              return_transition_function = True)
        
        # Correct/ajust the network data after flow generation
        return_dict_ajusted_data = fetch_ajust_ncorrect_flow_network_data(graph,
                                                                     raw_transport_times, 
                                                                     all_pairs, 
                                                                     return_multi_flow_dict)

        # Construct mfd_instance
        mfd_instance = MultiFlowDesagInstance(deepcopy(return_dict_ajusted_data["corr_graph"]), 
                                            deepcopy(return_dict_ajusted_data["aggregated_flow"]),
                                            deepcopy(return_dict_ajusted_data["transport_times"]), 
                                            deepcopy(return_dict_ajusted_data["pairs"]), 
                                            deepcopy(return_dict_ajusted_data["flow_values"]), 
                                            deepcopy(return_dict_ajusted_data["ls_transition_function"]),
                                            update_transport_time = True,
                                            update_transition_functions = True)
        
        # Saving the file
        np.save(os.path.join(dir_save_name_multiflow, 
                             "multi_flow_instance_"+str(num_instance)), 
                {"pairs":deepcopy(return_dict_ajusted_data["pairs"]),
                 "flow_values":deepcopy(return_dict_ajusted_data["flow_values"]),
                 "multi_flow":return_dict_ajusted_data["multi_flow"]})
        dict_instances[(num_instance, True, True)] = (mfd_instance, return_dict_ajusted_data["multi_flow"])
    
    # Save mfd_instances
    np.save(os.path.join(dir_save_name_mfd, 
                         "data_instances"), 
            dict_instances)
    


def main():
    test_names = {"versailles", "lieusaint"}
    test_name = "lieusaint"
    if test_name == "versailles":
        construct_real_instances (graph_nx_path_file = "data/real_data/original_graphs/versailles.gpickle", 
                                interest_points_file_path = "data/real_data/pre_processed/Versailles/points_versailles.txt",
                                dir_save_name_graph = "data/real_data/pre_processed/Versailles/",
                                dir_save_name_multiflow = "data/real_data/pre_processed/Versailles/multi_flow_instances/",
                                dir_save_name_mfd = "data/real_data/pre_processed/Versailles/",
                                nb_instances = 100,
                                nb_pairs = 15,
                                suffix_fname = "versailles",
                                car_size = 5,
                                min_fl = 1,
                                nb_max_draws_pairs = 300,
                                nb_it_print = None,
                                save_dir = "data/real_data/pre_processed/Versailles/",
                                matrix_representation = False,
                                generate_figure = [True, True])
    
    elif test_name == "lieusaint":
        construct_real_instances (graph_nx_path_file = "data/real_data/original_graphs/lieusaint.gpickle", 
                                interest_points_file_path = "data/real_data/pre_processed/LieuSaint/points_interet.txt",
                                dir_save_name_graph = "data/real_data/pre_processed/LieuSaint/",
                                dir_save_name_multiflow = "data/real_data/pre_processed/LieuSaint/multi_flow_instances/",
                                dir_save_name_mfd = "data/real_data/pre_processed/LieuSaint/",
                                nb_instances = 100,
                                nb_pairs = 15,
                                suffix_fname = "lieusaint",
                                car_size = 5,
                                min_fl = 1,
                                nb_max_draws_pairs = 300,
                                nb_it_print = 1,
                                save_dir = "data/real_data/pre_processed/LieuSaint/",
                                matrix_representation = False,
                                generate_figure = [True, True])

if __name__ == "__main__":
    main()