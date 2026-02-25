import sys

import os

import pickle

from itertools import product

import psutil

import numpy as np

import time

import matplotlib.pyplot as plt

import sys

import statistics

import pprint

from copy import deepcopy

import multiprocessing

import concurrent.futures

sys.path.append(os.getcwd())
from msmd.multi_flow_desag_RL_solver import MultiFlowDesagRLSolver
from experiments.test_utils import process_performances


# Function used to run experiments
def run_experiment (mfd_instance,
                    dict_results,
                    ind_instance,
                    test_infos,
                    original_multi_flow,
                    path_type_selector,
                    dict_params_rl_agent,
                    max_path_length, 
                    nb_max_tries,
                    nb_episodes,
                    max_nb_tries_find_path,
                    maximal_flow_amount,
                    reodering_pairs_policy_name,
                    opt_params,
                    pair_criteria, 
                    path_card_criteria,
                    ls_coeff,
                    graph_representation = "adjacency_matrix"):
    # Create an RL multi flow desaggregation solver and desagregate the multi flow
    solver = MultiFlowDesagRLSolver(mfd_instance = mfd_instance,
                                    path_selector_type = path_type_selector,
                                    dict_parameters = dict_params_rl_agent,
                                    max_path_length = max_path_length, 
                                    max_nb_it_episode = nb_max_tries,
                                    nb_episodes = nb_episodes,
                                    max_nb_tries_find_path = max_nb_tries_find_path,
                                    maximal_flow_amount = maximal_flow_amount,
                                    reodering_pairs_policy_name = reodering_pairs_policy_name,
                                    exclude_chosen_nodes = False,
                                    successor_selector_type = "exponential_decay",
                                    rl_data_init_type = "uniform",
                                    store_perfs_evol_path = None,
                                    ignore_conflicts = False,
                                    graph_representation = graph_representation,
                                    opt_params = opt_params)
    multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                        path_card_criteria,
                                                                        ls_coeff[0], ls_coeff[1], ls_coeff[2])
    process_performances(flow_vals_desagg, 
                        mfd_instance.original_flow_values, 
                        multi_flow_desag,
                        mfd_instance.original_aggregated_flow,
                        original_multi_flow,
                        mfd_instance.original_adj_mat,
                        mfd_instance.ideal_transport_times,
                        mfd_instance.pairs,
                        mfd_instance.original_transition_function,
                        solver,
                        dict_results,
                        ind_instance,
                        test_infos[-1],
                        opt_params = (test_infos[0], test_infos[1]),
                        graph_representation = graph_representation)


if __name__ == "__main__":
    print("Satring main.")
    #path_results = "results/simulated/MFDS_vs_RL/results_test/"+"results_rl_heuristics.npy"
    path_results = "results/"+"node_pairs_rl_heuristics.pickle"

    # Common parameters values
    max_path_length = 10000
    nb_max_tries = 50000
    max_nb_tries_find_path = 10
    maximal_flow_amount = 1
    pair_criteria = "max_remaining_flow_val"

    # Dict rl params
    nb_episodes = 31
    dict_params_rl_agent = {"ag_type":"LRI",
                            "lr":0.05,
                            "eps":None,
                            "opt_params":{"initial_actions_estimates":None}}
    coeffs_list = [(0.33, 0.33, 0.34)]

    # Test names
    ls_path_selector_types = ["rl_arc_based"]
    ls_path_card_criteria = ["one_for_each"]
    ls_learning_rates = [0.01, 0.025, 0.05, 0.075, 0.1]
    
    # Meta data
    res_key_metadata = ["nb_nodes", "nb_pairs", "id_instance",
                        ("path_type_selector", "path_card_criteria", 
                        "lr_rate", "coeffs_list")]
    res_value_metadata = ["flow_val_residue", "flow_residue", 
                          "multi_flow_residue", "prop_flow_support", 
                          "prop_shortest_paths", "transition_function_residue",
                          "reward"]
    

    debug = True
    multi_process = True
    if debug:
        multi_process = False
     
    nb_phys_cpus, nb_cpus = psutil.cpu_count(logical = False), psutil.cpu_count(logical = True)
    
    print("Nb of CPUs ", nb_phys_cpus, nb_cpus)
    
    jobs = []
    
    manager = multiprocessing.Manager()

    if multi_process:
        #nb_cpu_workers = nb_phys_cpus
        nb_cpu_workers = nb_cpus
    else:
        nb_cpu_workers = 1
    
    # Construction of the instances
    dict_node_pairs_instances = {}
    dir_constructed_instances_path = "data/simulated_data/node_pairs/"
    dir_list = os.listdir(dir_constructed_instances_path)
    for file_name in dir_list:
        if file_name[-4:] == ".npy":
            print("Load ", file_name)
            fl_splits = file_name.split("_")
            nb_nodes = int(fl_splits[2][0:2])
            pairs = int(fl_splits[3][0:2]) 
            dict_instances = np.load(dir_constructed_instances_path+file_name, 
                                     allow_pickle = True).flatten()[0]
            dict_node_pairs_instances[(nb_nodes, pairs)] = dict_instances

    # Main
    if debug:
        dict_results = {}
    else:
        dict_results = manager.dict()
        ls_args = [] 
    
    for nb_nodes, nb_pairs in dict_node_pairs_instances:
        dict_instances = dict_node_pairs_instances[(nb_nodes, nb_pairs)]
        for ind_instance, _, _ in dict_instances:
            print("Treating instance ", ind_instance)
            for path_type_selector in ls_path_selector_types:
                for path_card_criteria in ls_path_card_criteria:
                    for lr_rate in ls_learning_rates:
                        dict_params_rl_agent["lr"] = lr_rate
                        for coeff1, coeff2, coeff3 in coeffs_list:
                            # Test_infos
                            test_infos = (nb_nodes, nb_pairs,
                                          (path_type_selector, path_card_criteria, lr_rate, (coeff1, coeff2, coeff3)))

                            # Fetch instance
                            mfd_instance = deepcopy(dict_instances[(ind_instance, True, True)][0])
                            original_multi_flow = dict_instances[(ind_instance, True, True)][1]
                            
                            opt_params = {"penalty_init_val":0, 
                                        "decay_param":0.99}
                            
                            if path_card_criteria == "one_only":
                                reodering_pairs_policy_name = None
                            
                            elif path_card_criteria == "one_for_each":
                                reodering_pairs_policy_name = "remaining_max_flow"
                            
                            else:
                                print("Ordering unrecognized.")
                                sys.exit()

                            # Process additional/optional parameters
                            if coeff3 == 0:
                                mfd_instance.original_update_transition_functions =  False
                                mfd_instance.update_transition_functions = False

                            if debug:
                                print("path_selec:", path_type_selector,
                                    "-path_card:", path_card_criteria,
                                    "-lr_rate:", lr_rate,
                                    "-coeffs:", coeff1, coeff2, coeff3)
                                run_experiment (mfd_instance,
                                                dict_results,
                                                ind_instance,
                                                test_infos,
                                                original_multi_flow,
                                                path_type_selector,
                                                dict_params_rl_agent,
                                                max_path_length, 
                                                nb_max_tries,
                                                nb_episodes,
                                                max_nb_tries_find_path,
                                                maximal_flow_amount,
                                                reodering_pairs_policy_name,
                                                opt_params,
                                                pair_criteria, 
                                                path_card_criteria,
                                                (coeff1, coeff2, coeff3))
                            else:
                                ls_args.append((mfd_instance,
                                                dict_results,
                                                ind_instance,
                                                test_infos,
                                                original_multi_flow,
                                                path_type_selector,
                                                dict_params_rl_agent,
                                                max_path_length, 
                                                nb_max_tries,
                                                nb_episodes,
                                                max_nb_tries_find_path,
                                                maximal_flow_amount,
                                                reodering_pairs_policy_name,
                                                opt_params,
                                                pair_criteria, 
                                                path_card_criteria,
                                                (coeff1, coeff2, coeff3)))

    if not debug:
        nb_finished = 0
        print("Avancement ", 0, " %")
        with concurrent.futures.ProcessPoolExecutor(max_workers = nb_cpu_workers) as executor:
            results = [executor.submit(run_experiment, *args) for args in ls_args]
            for f in concurrent.futures.as_completed(results):
                f.result()
                nb_finished += 1
                print("Avancement ", round(100*nb_finished/len(results), 2), " %")
    
    # Save
    #  = ["path_type_selector", "path_card_criteria", "lr_rate", "coeffs_list"]
    with open(path_results, "wb") as handle:
        pickle.dump({"res_key_metadata":res_key_metadata,
                     "res_value_metadata":res_value_metadata, 
                     "data":deepcopy(dict(dict_results))},
                     handle,
                     protocol = pickle.HIGHEST_PROTOCOL)