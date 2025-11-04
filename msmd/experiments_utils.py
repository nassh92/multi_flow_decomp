import os

from itertools import product

import numpy as np

import time

import matplotlib.pyplot as plt

import sys

import statistics

import pprint

from copy import deepcopy

import os
sys.path.append(os.getcwd())
from utils.metrics import (transition_function_residue, 
                     flow_val_residue, 
                     flow_residue,
                     multi_flow_residue,
                     proportion_size_flow_support,
                     flow_proportion_shortest_paths)
from msmd.multi_flow_desag_general_solver import MultiFlowDesagSolver 
from msmd.multi_flow_desag_transf_solver import MultiFlowDesagSolverTransF
from msmd.multi_flow_desag_RL_solver import MultiFlowDesagRLSolver
from msmd.multi_flow_desag_instance_utils import construct_instances



def test_desaggregate_multi_flow (
        test_, 
        dict_params_rl_agent,
        mfd_instance,
        ind_instance,
        nb_episodes = 20,
        coeff1 = 0.33, coeff2 = 0.33, coeff3 = 0.34,
        pair_criteria = "max_remaining_flow_val",
        path_card_criteria = "one_only",
        maximal_flow_amount = 1,
        nb_max_tries = 1000000,
        max_path_length = 1000,
        max_nb_tries_find_path = 5,
        reodering_pairs_policy_name = None,
        successor_selector_type = "standard",
        rl_data_init_type = "flow_based",
        exclude_chosen_nodes = False,
        penalty_chosen_path = False,
        save_RL_dir = None,
        override_update_transition_function = [],
        ignore_conflicts = False):
    if test_ == "mfds_min_time":
        mfd_instance.reset_instance()
        # Create an multi flow desaggregation solver and desagregate the multi flow
        solver = MultiFlowDesagSolver(mfd_instance = mfd_instance,
                                    max_path_length = max_path_length, 
                                    total_nb_iterations = nb_max_tries,
                                    max_nb_tries_find_path = max_nb_tries_find_path,
                                    maximal_flow_amount = maximal_flow_amount,
                                    reodering_pairs_policy_name = reodering_pairs_policy_name,
                                    path_selector_type = "min_time_based",
                                    construct_trans_function = True,
                                    exclude_chosen_nodes = exclude_chosen_nodes,
                                    ignore_conflicts = ignore_conflicts)
        multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                        path_card_criteria)
        
    elif test_ == "mfds_max_capacity":
        # Reset Instance
        mfd_instance.reset_instance()
        # Create an multi flow desaggregation solver and desagregate the multi flow
        solver = MultiFlowDesagSolver(mfd_instance = mfd_instance,
                                    max_path_length = max_path_length, 
                                    total_nb_iterations = nb_max_tries,
                                    max_nb_tries_find_path = max_nb_tries_find_path,
                                    maximal_flow_amount = maximal_flow_amount,
                                    reodering_pairs_policy_name = reodering_pairs_policy_name,
                                    path_selector_type = "max_capacity_based",
                                    construct_trans_function = True,
                                    exclude_chosen_nodes = exclude_chosen_nodes,
                                    ignore_conflicts = ignore_conflicts)
        multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                        path_card_criteria)

    elif test_ == "mfds":
        # Reset Instance
        mfd_instance.reset_instance()
        # Create an multi flow desaggregation solver and desagregate the multi flow
        solver = MultiFlowDesagSolverTransF(mfd_instance = mfd_instance,
                                    max_path_length = max_path_length, 
                                    total_nb_iterations = nb_max_tries,
                                    max_nb_tries_find_path = max_nb_tries_find_path,
                                    maximal_flow_amount = maximal_flow_amount,
                                    reodering_pairs_policy_name = reodering_pairs_policy_name,
                                    path_selector_type = "trans_func_based",
                                    construct_trans_function = True,
                                    exclude_chosen_nodes = exclude_chosen_nodes,
                                    ignore_conflicts = ignore_conflicts)
        multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                        path_card_criteria)

    elif test_ == "mfds_random_path":
        # Reset Instance
        mfd_instance.reset_instance()
        # Create an multi flow desaggregation solver and desagregate the multi flow
        solver = MultiFlowDesagSolverTransF(mfd_instance = deepcopy(mfd_instance),
                                    max_path_length = max_path_length, 
                                    total_nb_iterations = nb_max_tries,
                                    max_nb_tries_find_path = max_nb_tries_find_path,
                                    maximal_flow_amount = maximal_flow_amount,
                                    reodering_pairs_policy_name = reodering_pairs_policy_name,
                                    path_selector_type = "random",
                                    construct_trans_function = True,
                                    exclude_chosen_nodes = exclude_chosen_nodes,
                                    ignore_conflicts = ignore_conflicts)
        multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                        path_card_criteria)
        
    elif test_ == "mfds_support_path":
        # Reset Instance
        mfd_instance.reset_instance()
        # Create an multi flow desaggregation solver and desagregate the multi flow
        solver = MultiFlowDesagSolverTransF(mfd_instance = deepcopy(mfd_instance),
                                    max_path_length = max_path_length, 
                                    total_nb_iterations = nb_max_tries,
                                    max_nb_tries_find_path = max_nb_tries_find_path,
                                    maximal_flow_amount = maximal_flow_amount,
                                    reodering_pairs_policy_name = reodering_pairs_policy_name,
                                    path_selector_type = "trans_func_support",
                                    construct_trans_function = True,
                                    exclude_chosen_nodes = exclude_chosen_nodes,
                                    ignore_conflicts = ignore_conflicts)
        multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                        path_card_criteria)
        
    elif test_ == "mfds_largest_flow":
        # Reset Instance
        mfd_instance.reset_instance()
        # Create an multi flow desaggregation solver and desagregate the multi flow
        solver = MultiFlowDesagSolverTransF(mfd_instance = deepcopy(mfd_instance),
                                    max_path_length = max_path_length, 
                                    total_nb_iterations = nb_max_tries,
                                    max_nb_tries_find_path = max_nb_tries_find_path,
                                    maximal_flow_amount = maximal_flow_amount,
                                    reodering_pairs_policy_name = reodering_pairs_policy_name,
                                    path_selector_type = "largest_flow_successor",
                                    construct_trans_function = True,
                                    exclude_chosen_nodes = exclude_chosen_nodes,
                                    ignore_conflicts = ignore_conflicts)
        multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                        path_card_criteria)
    
    elif test_ == "mfds_rl_node_based":
        # Reset instance
        mfd_instance.reset_instance()
        # Process additional/optional parameters
        if penalty_chosen_path:
            opt_params = {"penalty_init_val":0, 
                            "decay_param":0.99}
        else:
            opt_params = None
        
        adjusted_mfd_instance = deepcopy(mfd_instance)
        if coeff3 == 0:
            adjusted_mfd_instance.original_update_transition_functions =  False
            adjusted_mfd_instance.update_transition_functions = False
        
        if len(override_update_transition_function) > 0:
            adjusted_mfd_instance.original_update_transition_functions =  override_update_transition_function[0]
            adjusted_mfd_instance.update_transition_functions = override_update_transition_function[0]
        
        # Create an RL multi flow desaggregation solver and desagregate the multi flow
        store_perfs_evol_path = None if not save_RL_dir else save_RL_dir+"rl_evol_instance_"+str(ind_instance)+".npy"
        solver = MultiFlowDesagRLSolver(mfd_instance = adjusted_mfd_instance,
                                        path_selector_type = "rl_node_based",
                                        dict_parameters = dict_params_rl_agent,
                                        max_path_length = max_path_length, 
                                        max_nb_it_episode = nb_max_tries,
                                        nb_episodes = nb_episodes,
                                        max_nb_tries_find_path = max_nb_tries_find_path,
                                        maximal_flow_amount = maximal_flow_amount,
                                        reodering_pairs_policy_name = reodering_pairs_policy_name,
                                        exclude_chosen_nodes = exclude_chosen_nodes,
                                        successor_selector_type = successor_selector_type,
                                        rl_data_init_type = rl_data_init_type,
                                        store_perfs_evol_path = store_perfs_evol_path,
                                        ignore_conflicts = ignore_conflicts,
                                        opt_params = opt_params)
        multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                            path_card_criteria,
                                                                            coeff1, coeff2, coeff3)
        
    elif test_ == "mfds_rl_arc_based":
        # Reset instance
        mfd_instance.reset_instance()
        # Process additional/optional parameters
        if penalty_chosen_path:
            opt_params = {"penalty_init_val":0, 
                            "decay_param":0.99}
        else:
            opt_params = None
        
        adjusted_mfd_instance = deepcopy(mfd_instance)
        if coeff3 == 0:
            adjusted_mfd_instance.original_update_transition_functions =  False
            adjusted_mfd_instance.update_transition_functions = False
        
        if len(override_update_transition_function) > 0:
            adjusted_mfd_instance.original_update_transition_functions =  override_update_transition_function[0]
            adjusted_mfd_instance.update_transition_functions = override_update_transition_function[0]
        
        # Create an RL multi flow desaggregation solver and desagregate the multi flow
        store_perfs_evol_path = None if not save_RL_dir else save_RL_dir+"rl_evol_instance_"+str(ind_instance)+".npy"
        solver = MultiFlowDesagRLSolver(mfd_instance = adjusted_mfd_instance,
                                        path_selector_type = "rl_arc_based",
                                        dict_parameters = dict_params_rl_agent,
                                        max_path_length = max_path_length, 
                                        max_nb_it_episode = nb_max_tries,
                                        nb_episodes = nb_episodes,
                                        max_nb_tries_find_path = max_nb_tries_find_path,
                                        maximal_flow_amount = maximal_flow_amount,
                                        reodering_pairs_policy_name = reodering_pairs_policy_name,
                                        exclude_chosen_nodes = exclude_chosen_nodes,
                                        successor_selector_type = successor_selector_type,
                                        rl_data_init_type = rl_data_init_type,
                                        store_perfs_evol_path = store_perfs_evol_path,
                                        ignore_conflicts = ignore_conflicts,
                                        opt_params = opt_params)
        multi_flow_desag, flow_vals_desagg = solver.desagregate_multi_flow (pair_criteria, 
                                                                            path_card_criteria,
                                                                            coeff1, coeff2, coeff3)
        
    else:
        print("'test_' unrecognized.")
        sys.exit()

    return multi_flow_desag, flow_vals_desagg, solver


def process_data_MSMD_TransF_vs_MSMD_TF_RL (
                                     dict_instances,
                                     dict_params_rl_agent,
                                     nb_episodes = 20,
                                     coeff1 = 0.33, coeff2 = 0.33, coeff3 = 0.34,
                                     update_transport_time = True,
                                     update_transition_functions = True,
                                     pair_criteria = "max_remaining_flow_val",
                                     path_card_criteria = "one_only",
                                     maximal_flow_amount = 1,
                                     nb_max_tries = 1000000,
                                     max_path_length = 1000,
                                     max_nb_tries_find_path = 5,
                                     reodering_pairs_policy_name = None,
                                     successor_selector_type = "standard",
                                     rl_data_init_type = "flow_based",
                                     exclude_chosen_nodes = False,
                                     to_be_tested_list = ["mfds", "mfds_rl_node_based","mfds_rl_arc_based"], 
                                     penalty_chosen_path = False,
                                     save_RL_dir = None,
                                     override_update_transition_function = [],
                                     ignore_conflicts = False,
                                     exit_on_none = True):
    dict_result = {}
    for ind_instance, upd_transport_time, upd_transition_function in dict_instances:
        if upd_transport_time == update_transport_time and upd_transition_function == update_transition_functions:
            # Fetch the mfd instance
            mfd_instance, original_multi_flow = dict_instances[(ind_instance, upd_transport_time, upd_transition_function)]
            for test_ in to_be_tested_list:
                print("Case ", ind_instance, upd_transport_time, upd_transition_function, test_)
                
                multi_flow_desag, flow_vals_desagg, solver = test_desaggregate_multi_flow(
                    test_, 
                    dict_params_rl_agent,
                    mfd_instance,
                    ind_instance,
                    nb_episodes = nb_episodes,
                    coeff1 = coeff1, coeff2 = coeff2, coeff3 = coeff3,
                    pair_criteria = pair_criteria,
                    path_card_criteria = path_card_criteria,
                    maximal_flow_amount = maximal_flow_amount,
                    nb_max_tries = nb_max_tries,
                    max_path_length = max_path_length,
                    max_nb_tries_find_path = max_nb_tries_find_path,
                    reodering_pairs_policy_name = reodering_pairs_policy_name,
                    successor_selector_type = successor_selector_type,
                    rl_data_init_type = rl_data_init_type,
                    exclude_chosen_nodes = exclude_chosen_nodes,
                    penalty_chosen_path = penalty_chosen_path,
                    save_RL_dir = save_RL_dir,
                    override_update_transition_function = override_update_transition_function,
                    ignore_conflicts = ignore_conflicts
                )
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
                if prop_sp == None and exit_on_none:
                    print(multi_flow_desag)
                    sys.exit()
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
                dict_result[(ind_instance, upd_transport_time, upd_transition_function, test_)] = (flow_val_res, flow_res, m_flow_res, 
                                                                                                   prop_fsupp, prop_sp, trans_func_res)
    return dict_result