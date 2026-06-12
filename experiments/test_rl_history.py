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


def run_experiment (mfd_instance,
                    ind_instance,
                    dir_results,
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
                    successor_selector_type = "exponential_decay",
                    rl_data_init_type = "uniform",
                    graph_representation = "adjacency_matrix"):
    # Create an RL multi flow desaggregation solver and desaggregate the multi flow
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
                                    successor_selector_type = successor_selector_type,
                                    rl_data_init_type = rl_data_init_type,
                                    store_perfs_evol_path = None,
                                    ignore_conflicts = False,
                                    graph_representation = graph_representation,
                                    opt_params = opt_params)
    solver.desagregate_multi_flow (pair_criteria, 
                                   path_card_criteria,
                                   ls_coeff[0], ls_coeff[1], ls_coeff[2])
    path_results = dir_results+"/rl_history_"+str(ind_instance)
    with open(path_results, "wb") as handle:
        pickle.dump({"ind_instance":ind_instance,
                     "data":solver.performance_metrics_evol},
                     handle,
                     protocol = pickle.HIGHEST_PROTOCOL)



def basic_rl_heurs_lieu_saint_real_instances(constructed_instances_path,
                                             dir_results,
                                             graph_representation = "adjacency_matrix",
                                             debug = False,
                                             multi_process = True):
    print("Satring main.")
    print(constructed_instances_path)
    #print(path_results)
    
    # Common parameters values
    max_path_length = 10000
    nb_max_tries = 50000
    max_nb_tries_find_path = 20
    maximal_flow_amount = 1
    pair_criteria = "max_remaining_flow_val"

    # Dict rl params
    nb_episodes = 51
    print("Nb episodes ", nb_episodes)
    dict_params_rl_agent = {"ag_type":"LRI",
                            "lr":0.01,
                            "eps":None,
                            "opt_params":{"initial_actions_estimates":None}}
    coeff1, coeff2, coeff3 = 0.33, 0.33, 0.34
    
    # RL related params
    path_selector_type = "rl_arc_based"
    path_card_criteria = "one_only"
    
    # Successor selector
    successor_selector_type = "standard"
    rl_data_init_type = "uniform"

    # Meta data
    no_transition_function = False
    print("No trans func ", no_transition_function)
    #penalty_init_val, decay_param = 0, 0.99
    #print("Decay param ", decay_param)

    if debug:
        multi_process = False
     
    nb_phys_cpus, nb_cpus = psutil.cpu_count(logical = False), psutil.cpu_count(logical = True)
    
    print("Nb of CPUs ", nb_phys_cpus, nb_cpus)
    
    manager = multiprocessing.Manager()

    if multi_process:
        #nb_cpu_workers = nb_phys_cpus
        nb_cpu_workers = nb_cpus
    else:
        nb_cpu_workers = 1

    # Construction of the instances
    dict_instances = np.load(constructed_instances_path, 
                             allow_pickle = True).flatten()[0]
    print("Chargement du fichier terminé.")        

    # Main
    if debug:
        pass
    else:
        ls_args = []
    
    time.sleep(20)
    
    opt_params = {"penalty_init_val":0, 
                "decay_param":None,
                "reward_discount_type":"discount_by_cost",
                "penalize_circuits":False,
                "circuit_penalty_param":None,
                "graph_representation":graph_representation}
    
    if path_card_criteria == "one_only":
        reodering_pairs_policy_name = None
    
    elif path_card_criteria == "one_for_each":
        reodering_pairs_policy_name = "remaining_max_flow"
    
    else:
        print("Ordering unrecognized.")
        sys.exit()
    
    for ind_instance, _, _ in dict_instances:
        # Fetch instance
        mfd_instance = deepcopy(dict_instances[(ind_instance, True, True)][0])
        original_multi_flow = dict_instances[(ind_instance, True, True)][1]
        
        # Process additional/optional parameters
        if no_transition_function or coeff3 == 0:
            mfd_instance.original_update_transition_functions =  False
            mfd_instance.update_transition_functions = False
        
        if debug:
            run_experiment(mfd_instance,
                        ind_instance,
                        dir_results,
                        path_selector_type,
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
                        (coeff1, coeff2, coeff3),
                        successor_selector_type = successor_selector_type,
                        rl_data_init_type = rl_data_init_type,
                        graph_representation = graph_representation)
        else:
            ls_args.append((mfd_instance,
                        ind_instance,
                        dir_results,
                        path_selector_type,
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
                        (coeff1, coeff2, coeff3),
                        successor_selector_type,
                        rl_data_init_type,
                        graph_representation))

    if not debug:
        nb_finished = 0
        print("Avancement ", 0, " %")
        with concurrent.futures.ProcessPoolExecutor(max_workers = nb_cpu_workers) as executor:
            results = [executor.submit(run_experiment, *args) for args in ls_args]
            for f in concurrent.futures.as_completed(results):
                f.result()
                nb_finished += 1
                print("Avancement ", round(100*nb_finished/len(results), 2), " %")        
        

#Main function
def main():
    #constructed_instances_path = "data/real_data/pre_processed/Versailles/data_instances_versailles.npy"
    #constructed_instances_path = "data/real_data/pre_processed/data_instances_versailles.npy"
    #constructed_instances_path = "data/real_data/pre_processed/Versailles/data_instances_versailles_capacity.npy"
    constructed_instances_path = "data/real_data/Versailles/data_instances_versailles_capacity.npy"
    #path_results = "results/"+"results_versailles_rl_heuristics_capacity.pickle"
    dir_results = "results/" 
    basic_rl_heurs_lieu_saint_real_instances(constructed_instances_path,
                                             dir_results,
                                             graph_representation = "adjacency_list",
                                             debug = False,
                                             multi_process = True)


if __name__ == "__main__":
    main()