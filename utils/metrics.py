import math
import numpy as np
import sys
from copy import deepcopy
from shortest_path_solvers import DijkstraShortestPathsSolver
from graph_utils import adjacency_matrix_union

"""
offset because in the desaggregation heuristics, we add rows and columns to the matrix to represent the super source and the super destination

"""
############################################################################################################################################
#########################################################  SOLUTION LEVEL METRICS  #########################################################
############################################################################################################################################
def flow_val_residue (flow_values, orirignal_flow_values):
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


def flow_residue (multi_flow_desag, unattributed_flow, original_aggregated_flow, offset = 0):
    # Calculate the aggregated flow for the multi flow desagregation
    aggreg_flow = [[sum(multi_flow_desag[i][u][v] for i in range(len(multi_flow_desag)))+unattributed_flow[u][v]\
                                                                                            for v in range(len(multi_flow_desag[0])-offset)]\
                                                                                                for u in range(len(multi_flow_desag[0])-offset)]
    # Process the sum of the flow in all arcs
    if offset == 0:
        sum_orig_flow_agg = sum(sum(row) for row in original_aggregated_flow)
    else:
        sum_orig_flow_agg = sum(sum(row[:-offset]) for row in original_aggregated_flow[:-offset])
    # Process the sum of the difference between the two flows
    diff_flow = [abs(original_aggregated_flow[u][v] - aggreg_flow[u][v]) for u in range(len(aggreg_flow)) for v in range(len(aggreg_flow))]
    # Return the residue error
    return sum(diff_flow)/sum_orig_flow_agg


def multi_flow_residue (multi_flow_desag, original_multi_flow, original_aggregated_flow, offset = 0):
    # Process the sum of the flow in all arcs
    if offset == 0:
        sum_orig_flow_agg = sum(sum(row) for row in original_aggregated_flow)
    else:
        sum_orig_flow_agg = sum(sum(row[:-offset]) for row in original_aggregated_flow[:-offset])
    
    """for i in range(len(multi_flow_desag)):
        for u in range(len(multi_flow_desag[0])):
            for v in range(len(multi_flow_desag[0])):
                try:
                    a = original_multi_flow[i][u][v]
                except:
                    print("Length are ", len(multi_flow_desag), len(multi_flow_desag[0]))
                    print("Original multi flow ", i, u, v, " out of range")
                    sys.exit()
                try:
                    b = multi_flow_desag[i][u][v]
                except:
                    print("Multi flow desag ", i, u, v, " out of range")
                    sys.exit()"""
    # Process the difference between the real multi flow and the aggregated multif flow for each arc
    diff_per_arc = [sum(abs(original_multi_flow[i][u][v] - multi_flow_desag[i][u][v]) for i in range(len(original_multi_flow)))\
                                                                                                    for u in range(len(original_multi_flow[0]))\
                                                                                                        for v in range(len(original_multi_flow[0]))]
    return sum(diff_per_arc)/(2*sum_orig_flow_agg)


#######################################################   ADDITIONAL METRICS  #######################################################
def proportion_size_flow_support (multi_flow_desag, unattributed_flow, original_aggregated_flow, offset = 0):
    # Calculate size of support in the original aggregated flow
    len_support_org_aggregated_flow = sum(original_aggregated_flow[u][v] > 0 for u in range(len(original_aggregated_flow)-offset)\
                                                                                for v in range(len(original_aggregated_flow)-offset))
    # Calculate size of support in the new aggregated flow
    aggregated_flow = [[sum(multi_flow_desag[i][u][v] for i in range(len(multi_flow_desag)))+unattributed_flow[u][v]\
                                                                                                for v in range(len(multi_flow_desag[0])-offset)]\
                                                                                                    for u in range(len(multi_flow_desag[0])-offset)]
    len_support_aggregated_flow = sum(aggregated_flow[u][v] > 0 for u in range(len(aggregated_flow)) for v in range(len(aggregated_flow)))
    
    return len_support_aggregated_flow/len_support_org_aggregated_flow


#######################################################   SHORTEST PATH METRIC - FLOW arcs  #######################################################
def flow_proportion_shortest_paths (multi_flow_desag, unattributed_flow, adj_mat, transport_times, pairs, offset = 0):
    # Calculate the aggregated flow for the multi flow desagregation
    aggreg_flow = [[sum(multi_flow_desag[i][u][v] for i in range(len(multi_flow_desag)))+unattributed_flow[u][v]\
                                                                                            for v in range(len(multi_flow_desag[0])-offset)]\
                                                                                                for u in range(len(multi_flow_desag[0])-offset)]
    
    # Correct the graphs
    t_adj_mat = [[adj_mat[u][v] for v in range(len(adj_mat)-offset)] for u in range(len(adj_mat)-offset)]
    t_transport_times = [[transport_times[u][v] for v in range(len(transport_times)-offset)] for u in range(len(transport_times)-offset)]

    # Calculate sum of total flow
    sum_flow_agg = sum(sum(row) for row in aggreg_flow)
    if sum_flow_agg == 0:
        print("Will give a division by zero error.")
        return None

    # Process the union of the DAGs of shortest paths for each pairs
    sum_flow_agg_sp, union_mat = 0, None
    for id_pair in range(len(pairs)):
        source, destination = pairs[id_pair]
        dijkstra_solver = DijkstraShortestPathsSolver(source, t_adj_mat, t_transport_times, mode = "min_distance")
        dijkstra_solver.run_dijkstra()
        dijkstra_solver.construct_DAG_shortest_path(destination)
        union_mat = deepcopy(dijkstra_solver.dagsp) if id_pair == 0 else adjacency_matrix_union(union_mat, 
                                                                                                dijkstra_solver.dagsp)
    
    # Process the ratio between the sum of flow on the intersection of the DAGs over the sum of the flow on the all the arcs of the graph
    sum_flow_agg_sp += sum(aggreg_flow[u][v] for u in range(len(t_adj_mat)) for v in range(len(t_adj_mat)) if union_mat[u][v] == 1)

    return sum_flow_agg_sp/sum_flow_agg



#######################################################   Difference between constructed transition function   #######################################################
#######################################################               and real transition function             #######################################################
def transition_function_residue (original_transition_func, constructed_transition_func, original_aggregated_flow):
    # Sum of flow along all arcs of the graph
    sum_orig_flow_agg = sum(sum(row) for row in original_aggregated_flow)
    # Process the sum of the difference between the two transition function
    sum_diff_flow = sum(sum(abs(original_transition_func[arc][succ_arc] - constructed_transition_func[arc][succ_arc]) 
                                                                            for succ_arc in constructed_transition_func[arc]) 
                                                                                for arc in constructed_transition_func)
    # Return the transition function error
    return sum_diff_flow/sum_orig_flow_agg



##########################################################################################################################################
#######################################################   Instance level matrics   #######################################################
##########################################################################################################################################
def instance_flow_proportion_shortest_paths (adj_mat,
                                             original_aggregated_flow, 
                                             transport_times, 
                                             pairs):
    # Calculate sum of total flow
    sum_flow_agg = sum(sum(row) for row in original_aggregated_flow)
    if sum_flow_agg == 0:
        print("Will give a division by zero error.")
        return None
    
    # Calculate the adjacency matrix of the graph associated to the union of all 
    # shortest path DAGs associatd to each pair
    union_mat = 0
    for id_pair in range(len(pairs)):
        source, destination = pairs[id_pair]
        dijkstra_solver = DijkstraShortestPathsSolver(source, 
                                                      adj_mat, 
                                                      transport_times, 
                                                      mode = "min_distance")
        dijkstra_solver.run_dijkstra()
        dijkstra_solver.construct_DAG_shortest_path(destination)
        union_mat = deepcopy(dijkstra_solver.dagsp) if id_pair == 0 else adjacency_matrix_union(union_mat, 
                                                                                                dijkstra_solver.dagsp)
    
    # Process the ratio between the sum of flow on the intersection of the DAGs over the sum of the flow on the all the arcs of the graph
    sum_flow_agg_sp = sum(original_aggregated_flow[u][v] for u in range(len(adj_mat)) for v in range(len(adj_mat)) if union_mat[u][v] == 1)

    return sum_flow_agg_sp/sum_flow_agg