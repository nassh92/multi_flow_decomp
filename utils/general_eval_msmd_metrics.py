import math
import numpy as np
import sys
import os
from copy import deepcopy
sys.path.append(os.getcwd())
from utils.shortest_path_solvers import DijkstraShortestPathsSolver
from utils.graph_utils import sum_out_attributes, get_arcs, get_nodes, graph_union, has_arc, init_graph_arc_attribute_vals


"""
offset because in the desaggregation heuristics, we add rows and columns to the matrix to represent the super source and the super destination

"""
############################################################################################################################################
#########################################################  SOLUTION LEVEL METRICS  #########################################################
############################################################################################################################################
def flow_val_residue (flow_values, orirignal_flow_values):
    """
    Calculate the deviation error between the values of the decomposed flow and the values of the original flow.
    """
    # Process the sum of the flow in the orignal multiflow
    multi_flow_val = sum(orirignal_flow_values)
    i = 0

    while i < len(orirignal_flow_values):
        if orirignal_flow_values[i] < flow_values[i]:
            break
        i += 1
    if i < len(orirignal_flow_values):
        print("The value of the decomposed flow is superior to the value of the original flow.", i, orirignal_flow_values[i], flow_values[i])
        sys.exit()
    
    # Return the deviation error
    deviation_error = sum(abs(orirignal_flow_values[i] - flow_values[i]) for i in range(len(flow_values)))/multi_flow_val
    if deviation_error > 1:
        print("The total value of the flow is superior to 1.")
        sys.exit()
    
    return deviation_error


def flow_residue (multi_flow_desag, original_aggregated_flow, graph):
    """
    Calculate the residue error between the aggregated flow of the decomposed multiflow and the original aggregated flow.
    """
    # Get the arcs of the graph
    arcs_list = get_arcs(graph)
    # Calculate the aggregated flow for the multi flow desagregation
    aggreg_flow = init_graph_arc_attribute_vals(graph, init_val = 0)
    for u, v in arcs_list: aggreg_flow[u][v] = sum(multi_flow_desag[i][u][v] for i in range(len(multi_flow_desag)))
    # Process the sum of the flow in all arcs
    sum_orig_flow_agg = sum(original_aggregated_flow[u][v] for u, v in arcs_list)
    # Process the sum of the difference between the two flows
    diff_flow = [abs(original_aggregated_flow[u][v] - aggreg_flow[u][v]) for u, v in arcs_list]
    # Return the residue error
    return sum(diff_flow)/sum_orig_flow_agg


def multi_flow_residue (multi_flow_desag, original_multi_flow, original_aggregated_flow, graph):
    """
    Calculate the residue error between the decomposed multiflow and the original multiflow.
    """
    # Get the arcs of the graph
    arcs_list = get_arcs(graph)
    # Process the sum of the flow in all arcs
    sum_orig_flow_agg = sum(original_aggregated_flow[u][v] for u, v in arcs_list)
    # Process the difference between the real multi flow and the aggregated multif flow for each arc
    diff_per_arc = [sum(abs(original_multi_flow[i][u][v] - multi_flow_desag[i][u][v]) 
                                                            for i in range(len(original_multi_flow)))\
                                                                for u, v in arcs_list]
    return sum(diff_per_arc)/(2*sum_orig_flow_agg)


#######################################################   ADDITIONAL METRICS  #######################################################
def proportion_size_flow_support (multi_flow_desag, original_aggregated_flow, graph):
    """
    Calculate the proportion of the size of the support of the aggregated flow of the decomposed multiflow relative to the size of the support of the original aggregated flow.
    """
    # Get the arcs and the node of the graph
    arcs_list = get_arcs(graph)
    # Calculate size of support in the original aggregated flow
    len_support_org_aggregated_flow = sum(original_aggregated_flow[u][v] > 0 for u, v in arcs_list)
    # Calculate size of support in the new aggregated flow
    aggregated_flow = [sum(multi_flow_desag[i][u][v] for i in range(len(multi_flow_desag))) for u, v in arcs_list]
    len_support_aggregated_flow = sum(aggregated_flow_arc > 0 for aggregated_flow_arc in aggregated_flow)
    return len_support_aggregated_flow/len_support_org_aggregated_flow


#######################################################   SHORTEST PATH METRIC - FLOW arcs  #######################################################
def flow_proportion_shortest_paths (multi_flow_desag, graph, transport_times, pairs, matrix_representation = True):
    """
    Calculate the proportion of flow that is on the union of the DAGs of shortest paths for each pairs 
    in the desaggregated multiflow over the total flow associated to the desaggregated multiflow.
    """
    # Get the arcs and the node of the graph
    arcs_list = get_arcs(graph)
    # Calculate the aggregated flow for the multi flow desagregation
    aggreg_flow = init_graph_arc_attribute_vals(graph, init_val = 0)
    for u, v in arcs_list: aggreg_flow[u][v] = sum(multi_flow_desag[i][u][v] for i in range(len(multi_flow_desag)))

    # Calculate sum of total flow
    sum_flow_agg = sum(aggreg_flow[u][v] for u, v in arcs_list)
    if sum_flow_agg == 0:
        print("Will give a division by zero error.")
        return None

    # Process the union of the DAGs of shortest paths for each pairs
    for id_pair in range(len(pairs)):
        source, destination = pairs[id_pair]
        dijkstra_solver = DijkstraShortestPathsSolver(source, 
                                                      graph, 
                                                      transport_times, 
                                                      mode = "min_distance",
                                                      matrix_representation = matrix_representation)
        dijkstra_solver.run_dijkstra()
        dijkstra_solver.construct_DAG_shortest_path(destination)
        union_mat = deepcopy(dijkstra_solver.dagsp) if id_pair == 0 else graph_union(union_mat, 
                                                                                     dijkstra_solver.dagsp,
                                                                                     matrix_representation = matrix_representation)
    
    # Process the ratio between the sum of flow on the intersection of the DAGs over the sum of the flow on the all the arcs of the graph
    sum_flow_agg_sp = sum(aggreg_flow[u][v] for u, v in arcs_list if has_arc(union_mat, u, v))

    return sum_flow_agg_sp/sum_flow_agg



#######################################################   Difference between constructed transition function   #######################################################
#######################################################               and real transition function             #######################################################
def transition_function_residue (original_transition_func, constructed_transition_func, original_aggregated_flow, graph):
    """
    Calculate the residue error between the constructed transition function and the original transition function.
    """
    # Get the arcs and the node of the graph
    arcs_list = get_arcs(graph)
    # Sum of flow along all arcs of the graph
    sum_orig_flow_agg = sum(original_aggregated_flow[u][v] for u, v in arcs_list)
    # Process the sum of the difference between the two transition functions (the constructed one and the original one)
    sum_diff_flow = sum(sum(abs(original_transition_func[arc][succ_arc] - constructed_transition_func[arc][succ_arc]) 
                                                                            for succ_arc in constructed_transition_func[arc]) 
                                                                                for arc in constructed_transition_func)
    # Return the transition function error
    return sum_diff_flow/sum_orig_flow_agg



##########################################################################################################################################
#######################################################   Instance level matrics   #######################################################
##########################################################################################################################################
def instance_flow_proportion_shortest_paths (graph,
                                             original_aggregated_flow, 
                                             transport_times, 
                                             pairs,
                                             matrix_representation = True):
    """
    Calculate the proportion of flow that is on the union of the DAGs of shortest paths for each pairs over the total flow.
    """
    # Get the arcs of the graph
    arcs = get_arcs(graph)

    # Calculate sum of total flow
    sum_flow_agg = sum(original_aggregated_flow[u][v] for u, v in arcs)
    if sum_flow_agg == 0:
        print("Will give a division by zero error.")
        return None
    
    # Calculate the adjacency matrix of the graph associated to the union of all 
    # shortest path DAGs associatd to each pair
    for id_pair in range(len(pairs)):
        source, destination = pairs[id_pair]
        dijkstra_solver = DijkstraShortestPathsSolver(source, 
                                                      graph, 
                                                      transport_times, 
                                                      mode = "min_distance",
                                                      matrix_representation = matrix_representation)
        dijkstra_solver.run_dijkstra()
        dijkstra_solver.construct_DAG_shortest_path(destination)
        union_mat = deepcopy(dijkstra_solver.dagsp) if id_pair == 0 else graph_union (union_mat, 
                                                                                      dijkstra_solver.dagsp,
                                                                                      matrix_representation = matrix_representation)
    
    # Process the ratio between the sum of flow on the intersection of the DAGs over the sum of the flow on the all the arcs of the graph
    sum_flow_agg_sp = sum(original_aggregated_flow[u][v] for u, v in arcs if has_arc(union_mat, u, v))
    return sum_flow_agg_sp/sum_flow_agg