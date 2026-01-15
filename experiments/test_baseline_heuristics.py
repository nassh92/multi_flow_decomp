import os

import numpy as np

import pickle

import matplotlib.pyplot as plt

import sys

from copy import deepcopy

sys.path.append(os.getcwd())
from utils.metrics import (transition_function_residue, 
                     flow_val_residue, 
                     flow_residue,
                     multi_flow_residue,
                     proportion_size_flow_support,
                     flow_proportion_shortest_paths)
from msmd.multi_flow_desag_instance_utils import construct_instances
from msmd.multi_flow_desag_general_solver import MultiFlowDesagSolver 
from msmd.multi_flow_desag_transf_solver import MultiFlowDesagSolverTransF


def baseline_heurs_simulated_instances():
    #print("Current directory : ", os.getcwd())
    # Directories
    dir_name_graph_instance = "instance_generation/instances/capacity/"
    dir_name_multi_flow_instance = "multi_flow_generation/transition_function_instances/"
    path_results = "results/simulated/MFDS_vs_RL/temp/"+"results_baseline_heuristics_mock.pickle"

    print("Instances construction ")
    # Construction of the instances
    dict_instances = construct_instances (
                        dir_name_graph_instance = dir_name_graph_instance, 
                        dir_name_multi_flow_instance = dir_name_multi_flow_instance,
                        nb_instances = 100,
                        ls_update_transport_time = [True],
                        ls_update_transition_functions = [True])

    # Common parameters values
    max_path_length = 10000
    nb_max_tries = 50000
    max_nb_tries_find_path = 10
    maximal_flow_amount = 1

    # Fixed parameters (for now)
    reodering_pairs_policy_name = None
    pair_criteria = "max_remaining_flow_val"
    path_card_criteria = "one_only"

    # Test names
    test_names = ["min_time", 
                "max_capacity", 
                "trans_func", 
                "random"]

    # Main loop
    dict_results = {}
    for ind_instance, _, _ in dict_instances:
        print("Treating instance ", ind_instance)
        # Fetch instance
        mfd_instance = dict_instances[(ind_instance, True, True)][0]
        original_multi_flow = dict_instances[(ind_instance, True, True)][1]

        # Test the tests
        for test in test_names:
            if test == "min_time":
                # Create an multi flow desaggregation solver and desagregate the multi flow
                solver = MultiFlowDesagSolver(mfd_instance = deepcopy(mfd_instance),
                                            max_path_length = max_path_length, 
                                            total_nb_iterations = nb_max_tries,
                                            max_nb_tries_find_path = max_nb_tries_find_path,
                                            maximal_flow_amount = maximal_flow_amount,
                                            reodering_pairs_policy_name = reodering_pairs_policy_name,
                                            path_selector_type = "min_time_based",
                                            construct_trans_function = True,
                                            exclude_chosen_nodes = False,
                                            ignore_conflicts = False)
                multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                                    path_card_criteria)
            
            elif test == "max_capacity":
                # Create an multi flow desaggregation solver and desagregate the multi flow
                solver = MultiFlowDesagSolver(mfd_instance = deepcopy(mfd_instance),
                                            max_path_length = max_path_length, 
                                            total_nb_iterations = nb_max_tries,
                                            max_nb_tries_find_path = max_nb_tries_find_path,
                                            maximal_flow_amount = maximal_flow_amount,
                                            reodering_pairs_policy_name = reodering_pairs_policy_name,
                                            path_selector_type = "max_capacity_based",
                                            construct_trans_function = True,
                                            exclude_chosen_nodes = False,
                                            ignore_conflicts = False)
                multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                                    path_card_criteria)

            elif test == "trans_func":
                # Create an multi flow desaggregation solver and desagregate the multi flow
                solver = MultiFlowDesagSolverTransF(mfd_instance = deepcopy(mfd_instance),
                                            max_path_length = max_path_length, 
                                            total_nb_iterations = nb_max_tries,
                                            max_nb_tries_find_path = max_nb_tries_find_path,
                                            maximal_flow_amount = maximal_flow_amount,
                                            reodering_pairs_policy_name = reodering_pairs_policy_name,
                                            path_selector_type = "trans_func_based",
                                            construct_trans_function = True,
                                            exclude_chosen_nodes = False,
                                            ignore_conflicts = False)
                multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                                    path_card_criteria)

            elif test == "random":
                # Create an multi flow desaggregation solver and desagregate the multi flow
                solver = MultiFlowDesagSolverTransF(mfd_instance = deepcopy(mfd_instance),
                                            max_path_length = max_path_length, 
                                            total_nb_iterations = nb_max_tries,
                                            max_nb_tries_find_path = max_nb_tries_find_path,
                                            maximal_flow_amount = maximal_flow_amount,
                                            reodering_pairs_policy_name = reodering_pairs_policy_name,
                                            path_selector_type = "random",
                                            construct_trans_function = True,
                                            exclude_chosen_nodes = False,
                                            ignore_conflicts = False)
                multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                                    path_card_criteria)
            
            else:
                print("Error.")
                sys.exit()
        

            unattributed_flow = [[0 for v in range(len(mfd_instance.aggregated_flow))] 
                                                        for u in range(len(mfd_instance.aggregated_flow))]
            # Process the metrics and store them in 'dict_result'
            flow_val_res = flow_val_residue (flow_vals_desagg, 
                                            mfd_instance.original_flow_values)
            flow_res = flow_residue (multi_flow_desag, 
                                    unattributed_flow, 
                                    mfd_instance.original_aggregated_flow)
            m_flow_res = multi_flow_residue (multi_flow_desag, 
                                            original_multi_flow, 
                                            mfd_instance.original_aggregated_flow)
            prop_fsupp = proportion_size_flow_support (multi_flow_desag, 
                                                    unattributed_flow, 
                                                    mfd_instance.original_aggregated_flow)
            prop_sp = flow_proportion_shortest_paths (multi_flow_desag, 
                                                    unattributed_flow, 
                                                    mfd_instance.original_adj_mat, 
                                                    mfd_instance.ideal_transport_times, 
                                                    mfd_instance.pairs)
            """if prop_sp == None and exit_on_none:
                print(multi_flow_desag)
                sys.exit()"""
            trans_func_res = transition_function_residue (mfd_instance.original_transition_function, 
                                                        solver.constructed_transition_function, 
                                                        mfd_instance.original_aggregated_flow)
            print("Proportion of support arcs  ", prop_fsupp)
            print("Proportion of flow val residue ", flow_val_res)
            print("Proportion of flow residue  ", flow_res)
            print("Proportion of multi flow residue  ", m_flow_res)
            print("Proportion of trans func residue  ", trans_func_res)
            print("Original flow values ", mfd_instance.original_flow_values)
            print("Desaggregated Flow values ", flow_vals_desagg)
            #time.sleep(0.5)
            dict_results[(ind_instance, test)] = (flow_val_res, flow_res, m_flow_res, 
                                                prop_fsupp, prop_sp, trans_func_res)
            
    # Save
    res_key_metadata = test_names
    res_value_metadata = ["flow_val_residue", "flow_residue", 
                        "multi_flow_residue", "prop_flow_support", 
                        "prop_shortest_paths", "transition_function_residue"]

    with open(path_results, "wb") as handle:
        pickle.dump({"res_key_metadata":res_key_metadata,
                    "res_value_metadata":res_value_metadata, 
                    "data":deepcopy(dict(dict_results))},
                    handle,
                    protocol = pickle.HIGHEST_PROTOCOL)    


################################################################################################
def baseline_heurs_real_instances(constructed_instances_path,
                                  path_results):
    #print("Current directory : ", os.getcwd())
    # Construction of the instances
    dict_instances = np.load(constructed_instances_path, 
                             allow_pickle = True).flatten()[0]

    # Common parameters values
    max_path_length = 10000
    nb_max_tries = 50000
    max_nb_tries_find_path = 10
    maximal_flow_amount = 1

    # Fixed parameters (for now)
    reodering_pairs_policy_name = None
    pair_criteria = "max_remaining_flow_val"
    path_card_criteria = "one_only"

    # Test names
    """test_names = ["min_time", 
                "max_capacity", 
                "trans_func", 
                "random"]"""
    test_names = ["trans_func"]

    # Main loop
    dict_results = {}
    for ind_instance, _, _ in dict_instances:
        print("Treating instance ", ind_instance)
        # Fetch instance
        mfd_instance = dict_instances[(ind_instance, True, True)][0]
        original_multi_flow = dict_instances[(ind_instance, True, True)][1]

        # Test the tests
        for test in test_names:
            print("Test name ", test)
            if test == "min_time":
                # Create an multi flow desaggregation solver and desagregate the multi flow
                solver = MultiFlowDesagSolver(mfd_instance = deepcopy(mfd_instance),
                                            max_path_length = max_path_length, 
                                            total_nb_iterations = nb_max_tries,
                                            max_nb_tries_find_path = max_nb_tries_find_path,
                                            maximal_flow_amount = maximal_flow_amount,
                                            reodering_pairs_policy_name = reodering_pairs_policy_name,
                                            path_selector_type = "min_time_based",
                                            construct_trans_function = True,
                                            exclude_chosen_nodes = False,
                                            ignore_conflicts = False)
                multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                                    path_card_criteria)
            
            elif test == "max_capacity":
                # Create an multi flow desaggregation solver and desagregate the multi flow
                solver = MultiFlowDesagSolver(mfd_instance = deepcopy(mfd_instance),
                                            max_path_length = max_path_length, 
                                            total_nb_iterations = nb_max_tries,
                                            max_nb_tries_find_path = max_nb_tries_find_path,
                                            maximal_flow_amount = maximal_flow_amount,
                                            reodering_pairs_policy_name = reodering_pairs_policy_name,
                                            path_selector_type = "max_capacity_based",
                                            construct_trans_function = True,
                                            exclude_chosen_nodes = False,
                                            ignore_conflicts = False)
                multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                                    path_card_criteria)

            elif test == "trans_func":
                # Create an multi flow desaggregation solver and desagregate the multi flow
                solver = MultiFlowDesagSolverTransF(mfd_instance = deepcopy(mfd_instance),
                                            max_path_length = max_path_length, 
                                            total_nb_iterations = nb_max_tries,
                                            max_nb_tries_find_path = max_nb_tries_find_path,
                                            maximal_flow_amount = maximal_flow_amount,
                                            reodering_pairs_policy_name = reodering_pairs_policy_name,
                                            path_selector_type = "trans_func_based",
                                            construct_trans_function = True,
                                            exclude_chosen_nodes = False,
                                            ignore_conflicts = False)
                multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                                    path_card_criteria)

            elif test == "random":
                # Create an multi flow desaggregation solver and desagregate the multi flow
                solver = MultiFlowDesagSolverTransF(mfd_instance = deepcopy(mfd_instance),
                                            max_path_length = max_path_length, 
                                            total_nb_iterations = nb_max_tries,
                                            max_nb_tries_find_path = max_nb_tries_find_path,
                                            maximal_flow_amount = maximal_flow_amount,
                                            reodering_pairs_policy_name = reodering_pairs_policy_name,
                                            path_selector_type = "random",
                                            construct_trans_function = True,
                                            exclude_chosen_nodes = False,
                                            ignore_conflicts = False)
                multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                                    path_card_criteria)
            
            else:
                print("Error.")
                sys.exit()
        
            unattributed_flow = [[0 for v in range(len(mfd_instance.aggregated_flow))] 
                                                        for u in range(len(mfd_instance.aggregated_flow))]
            # Process the metrics and store them in 'dict_result'
            flow_val_res = flow_val_residue (flow_vals_desagg, 
                                            mfd_instance.original_flow_values)
            flow_res = flow_residue (multi_flow_desag, 
                                    unattributed_flow, 
                                    mfd_instance.original_aggregated_flow)
            m_flow_res = multi_flow_residue (multi_flow_desag, 
                                            original_multi_flow, 
                                            mfd_instance.original_aggregated_flow)
            prop_fsupp = proportion_size_flow_support (multi_flow_desag, 
                                                    unattributed_flow, 
                                                    mfd_instance.original_aggregated_flow)
            prop_sp = flow_proportion_shortest_paths (multi_flow_desag, 
                                                    unattributed_flow, 
                                                    mfd_instance.original_adj_mat, 
                                                    mfd_instance.ideal_transport_times, 
                                                    mfd_instance.pairs)
            """if prop_sp == None and exit_on_none:
                print(multi_flow_desag)
                sys.exit()"""
            trans_func_res = transition_function_residue (mfd_instance.original_transition_function, 
                                                        solver.constructed_transition_function, 
                                                        mfd_instance.original_aggregated_flow)
            print("Proportion of support arcs  ", prop_fsupp)
            print("Proportion of flow val residue ", flow_val_res)
            print("Proportion of flow residue  ", flow_res)
            print("Proportion of multi flow residue  ", m_flow_res)
            print("Proportion of trans func residue  ", trans_func_res)
            print("Original flow values ", mfd_instance.original_flow_values)
            print("Desaggregated Flow values ", flow_vals_desagg)
            #time.sleep(0.5)
            dict_results[(ind_instance, test)] = (flow_val_res, flow_res, m_flow_res, 
                                                prop_fsupp, prop_sp, trans_func_res)
            
    # Save
    res_key_metadata = test_names
    res_value_metadata = ["flow_val_residue", "flow_residue", 
                        "multi_flow_residue", "prop_flow_support", 
                        "prop_shortest_paths", "transition_function_residue"]

    with open(path_results, "wb") as handle:
        pickle.dump({"res_key_metadata":res_key_metadata,
                    "res_value_metadata":res_value_metadata, 
                    "data":deepcopy(dict(dict_results))},
                    handle,
                    protocol = pickle.HIGHEST_PROTOCOL)


#Main function
def main():
    test_names = {"simulated_instances", "real_instances"}
    test_name = "real_instances"

    if test_name == "simulated_instances":
        baseline_heurs_simulated_instances()

    elif test_name == "real_instances":
        # Directories
        #constructed_instances_path = "multi_flow_generation_wei/data/data_instances.npy"
        constructed_instances_path = "data/pre_processed/LieuSaint/data_instances.npy"
        #path_results = "results/simulated/MFDS_vs_RL/results_test/"+"results_rl_heuristics.npy"
        #path_results = "results/"+"results_versailles_heuristics.pickle"
        path_results = "results/"+"results_lieusaint2_heuristics.pickle"
        baseline_heurs_real_instances(constructed_instances_path,
                                      path_results)

    elif test_name not in test_names:
        print("Test name is unrecognized.")
        sys.exit()


if __name__ == "__main__":
    main()