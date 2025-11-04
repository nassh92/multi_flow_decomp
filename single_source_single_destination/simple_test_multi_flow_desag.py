from copy import deepcopy

from itertools import product

import os

import sys

sys.path.append(os.getcwd())
from utils.shortest_path_solvers import MODES, PATH_TYPES

from single_source_single_destination.multi_flow_desagregation_heuristics import MultiFlowDesagSolver


def save_simple_instance_specific_case(adj_mat, aggregated_flow, transport_times, pairs, flow_values, 
                                       update_transport_time, path_type, mode, 
                                       dir_save_name, show = False, seed = 42):
    mf_decomp = MultiFlowDesagSolver(adj_mat = adj_mat, 
                                     aggregated_flow = aggregated_flow,
                                     transport_times = transport_times, 
                                     pairs = pairs, 
                                     flow_values = flow_values,
                                     update_transport_time = update_transport_time)
        
    multi_flow, flow_vals = mf_decomp.heuristic_multi_flow_desagregation (path_type = path_type, 
                                                                          mode = mode, 
                                                                          seed = seed,
                                                                          keep_logs=True,
                                                                          dir_save_name = dir_save_name,
                                                                          show = show)


def process_dir_save_name(update_transport_time, mode, path_type):
    dir_name1 = "update_time/" if update_transport_time else "not_update_time/"
    dir_name2 = mode+"_"+path_type+"/"
    return dir_name1+dir_name2


def save_simple_instances(adj_mat, aggregated_flow, transport_times, pairs, flow_values, dir_save_name, seed = 42):
    # product(vals of 'update_transport_time'  x  vals of 'path_type'   x   vals of 'mode')
    product_cases = list(product([False, True], PATH_TYPES, MODES))
    for update_transport_time, path_type, mode in product_cases:
        dir_path = dir_save_name+process_dir_save_name(update_transport_time, mode, path_type)
        save_simple_instance_specific_case(adj_mat, aggregated_flow, transport_times, pairs, flow_values, 
                                           update_transport_time = update_transport_time, 
                                           path_type = path_type, 
                                           mode = mode, 
                                           dir_save_name = dir_path, 
                                           show = False, 
                                           seed = seed)


def process_simple_instances(dir_save_names, seed = 42):
    # Save for "Rapport master, Exemple de la Figure 2"
    print("Treating Case : ", dir_save_names[0])
    adj_mat = [[0, 1, 1, 0, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 1, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 1],
               [1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]   
    aggregated_flow = [[0, 1, 2, 0, 0, 0],
                       [0, 0, 1, 2, 0, 0],
                       [0, 2, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 3],
                       [3, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]]
    transport_times = [[0 if i==j else 1 if adj_mat[i][j]==1 else float("inf") for j in range(len(adj_mat))] for i in range(len(adj_mat))]
    pairs = [(0,3)]
    flow_values = [3]
    save_simple_instances(adj_mat, aggregated_flow, transport_times, pairs, flow_values, dir_save_names[0], seed = seed)

    # Save for "Rapport master, Exemple de la Figure 2"
    print("Treating Case : ", dir_save_names[1])
    adj_mat = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    aggregated_flow = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    transport_times = [[0 if i==j else 1 if adj_mat[i][j]==1 else float("inf") for j in range(len(adj_mat))] for i in range(len(adj_mat))]
    pairs = [(0,7), (0,8)]
    flow_values = [1, 1]
    save_simple_instances(adj_mat, aggregated_flow, transport_times, pairs, flow_values, dir_save_names[1], seed = seed)

    # Save for "Rapport master, Exemple de la Figure 3", pairs : (1, 4), (2, 1) et (2, 4) (numérotation 1 à 4)
    print("Treating Case : ", dir_save_names[2])
    adj_mat = [[0, 0, 0, 1, 0, 1],
               [1, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 1],
               [1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]  
    aggregated_flow = [[0, 0, 0, 1, 0, 1],
                       [1, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 2],
                       [1, 2, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]]
    transport_times = [[0 if i==j else 1 if adj_mat[i][j]==1 else float("inf") for j in range(len(adj_mat))] for i in range(len(adj_mat))]
    pairs = [(0, 3), (1, 0), (1, 3)]
    flow_values = [1, 1, 1]
    save_simple_instances(adj_mat, aggregated_flow, transport_times, pairs, flow_values, dir_save_names[2], seed = seed)

    # Save for "Rapport master, Exemple de la Figure 4"
    print("Treating Case : ", dir_save_names[3])
    adj_mat = [[0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0]] 
    aggregated_flow = [[0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    transport_times = [[0 if i==j else 1 if adj_mat[i][j]==1 else float("inf") for j in range(len(adj_mat))] for i in range(len(adj_mat))]
    pairs = [(0, 4), (5, 2)]
    flow_values = [1, 1]
    save_simple_instances(adj_mat, aggregated_flow, transport_times, pairs, flow_values, dir_save_names[3], seed = seed)


if __name__ == "__main__":
    process_simple_instances(dir_save_names = ["simple_test_heursitics/decomposition_de_flots_nass_corr_exemple_de_la_Figure_5/",
                                               "simple_test_heursitics/rapport_master_exemple_de_la_Figure_2/",
                                               "simple_test_heursitics/rapport_master_exemple_de_la_Figure_3/",
                                               "simple_test_heursitics/rapport_master_exemple_de_la_Figure_4/"], 
                             seed = 76)
    # 42 63 67 76