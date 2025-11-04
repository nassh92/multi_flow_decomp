import os

from itertools import product

import numpy as np

import time

import matplotlib.pyplot as plt

import sys

import statistics

import pprint

import os
sys.path.append(os.getcwd())
from single_source_single_destination.multi_flow_desagregation_heuristics import (MultiFlowDesagSolverSSSD, 
                                                 MultiFlowDesagSolverMSMD,
                                                 MultiFlowDesagSolverMSMDTransFunc,
                                                 return_nb_equal_flow_val, 
                                                 return_nb_total_it)

from utils.shortest_path_solvers import PATH_TYPES, MODES

from utils.metrics import (flow_val_residue, 
                     flow_residue, 
                     multi_flow_residue, 
                     proportion_size_flow_support, 
                     flow_proportion_shortest_paths, 
                     flow_val_proportion_shortest_paths)

# Suppression of flow_val_proportion_shortest_paths

from utils.pre_process_utils import read_ntreat_instance

from single_source_single_destination.run_experiences_heurs_utils import (save_performances_metrics, 
                                         display_perfs_simulated_instances_SSSD, 
                                         display_perfs_simulated_instances_MSMD, 
                                         display_multi_flow_residues_perfs_SSSD, 
                                         display_multi_flow_residues_perfs_MSMD, 
                                         process_statistics, 
                                         test_correlation, 
                                         process_statistics_real, 
                                         display_comparison_SSSD_MSMD,
                                         compare_ndispplay_pairs_criterias_perfs_simul_instances_MSMD)

from single_source_single_destination.run_experiences_heurs_utils import (compare_MSMD_vs_MSMD_Trans_Func,
                                         compare_MSMD_Trans_Func_vs_MSMD_Trans_Func_MulP)



######################################################################################################################################
#######################################   This function evaluates the simulated instances of the restore-problem   ###################
######################################################################################################################################
def eval_simulated_instances(dir_name_graph_instance, 
                             dir_name_multi_flow_instance, 
                             nb_instances,
                             test_cases,
                             alg_name_info = "SSSD",
                             maximal_flow_amount = float("inf"),
                             trans_func = False):
    alg_name = alg_name_info if isinstance(alg_name_info, str) else alg_name_info[0]
    # File names of the instances
    dict_result = {}
    for i in range(nb_instances):
        # Read the graph
        print("-------------------------------------------------------------------------------------")
        print("Treating instance ", i)
        # Read the (complete) instance
        if alg_name == "SSSD":
            complete_instance = read_ntreat_instance (dir_name_graph_instance = dir_name_graph_instance, 
                                                      dir_name_multi_flow_instance = dir_name_multi_flow_instance,
                                                      id_instance = i,
                                                      trans_func = trans_func)
        elif alg_name == "MSMD":
            complete_instance = read_ntreat_instance (dir_name_graph_instance = dir_name_graph_instance, 
                                                      dir_name_multi_flow_instance = dir_name_multi_flow_instance,
                                                      id_instance = i,
                                                      add_supsource_supdestination = False,
                                                      trans_func = trans_func)

        # Unpack the instance
        adj_mat = complete_instance["adj_mat"]
        aggregated_flow = complete_instance["aggregated_flow"]
        transport_times = complete_instance["transport_times"]
        pairs = complete_instance["pairs"]
        flow_values = complete_instance["flow_values"]
        aggregated_flow = complete_instance["aggregated_flow"]
        multi_flow = complete_instance["multi_flow"]
        
        # Desaggregate the flow for each cas in 'test_cases' case and return the result in dict 'dict_result'
        for update_transport_time, path_type, mode, pair_criteria in test_cases:
            # Create the solver
            if alg_name == "SSSD":
                # Create an instance of 'MultiFlowDesagSolverSSSD'
                mf_decomp = MultiFlowDesagSolverSSSD(adj_mat = adj_mat, 
                                                aggregated_flow = aggregated_flow,
                                                transport_times = transport_times, 
                                                pairs = pairs,
                                                flow_values = flow_values,
                                                update_transport_time = update_transport_time,
                                                maximal_flow_amount = maximal_flow_amount)
                #  Use to desaggregate the multiflow with a call to method 'heuristic_multi_flow_desagregation'
                multi_flow_desag, flow_vals_desagg = mf_decomp.heuristic_multi_flow_desagregation (path_type = path_type,
                                                                                                   mode = mode)
                unattributed_flow = mf_decomp.unattributed_flow
            
            elif alg_name == "MSMD":
                if isinstance(alg_name_info, str):
                    # Create an instance of 'MultiFlowDesagSolverMSMD'
                    mf_decomp = MultiFlowDesagSolverMSMD(adj_mat = adj_mat, 
                                                    aggregated_flow = aggregated_flow,
                                                    transport_times = transport_times, 
                                                    pairs = pairs,
                                                    flow_values = flow_values,
                                                    update_transport_time = update_transport_time,
                                                    maximal_flow_amount = maximal_flow_amount)
                    #  Use to desaggregate the multiflow with a call to method 'heuristic_multi_flow_desagregation'
                    multi_flow_desag, flow_vals_desagg = mf_decomp.heuristic_multi_flow_desagregation (path_type = path_type,
                                                                                                       mode = mode,
                                                                                                       pair_criteria = pair_criteria)
                    unattributed_flow = mf_decomp.unattributed_flow
                
                elif isinstance(alg_name_info, list) and alg_name_info[1] == "trans_func":
                    # Create an instance of 'MultiFlowDesagSolverMSMDTransFunc'
                    mf_decomp = MultiFlowDesagSolverMSMDTransFunc(adj_mat = adj_mat, 
                                                                  aggregated_flow = aggregated_flow, 
                                                                  transport_times = transport_times, 
                                                                  pairs = pairs, 
                                                                  flow_values = flow_values, 
                                                                  ls_transition_function = complete_instance["transition_functions"], 
                                                                  update_transport_time = update_transport_time)
                    #  Use to desaggregate the multiflow with a call to method 'heuristic_multi_flow_desagregation'
                    multi_flow_desag, flow_vals_desagg = mf_decomp.heuristic_multi_flow_desagregation (pair_criteria = pair_criteria)
                    unattributed_flow = mf_decomp.unattributed_flow
            
            # Process the metrics and store them in 'dict_result'
            flow_val_res = flow_val_residue (flow_vals_desagg, flow_values)
            flow_res = flow_residue (multi_flow_desag, unattributed_flow, aggregated_flow, 
                                     offset = 2 if alg_name == "SSSD" else 0)
            m_flow_res = multi_flow_residue (multi_flow_desag, multi_flow, aggregated_flow, 
                                             offset = 2 if alg_name == "SSSD" else 0)
            prop_fsupp = proportion_size_flow_support (multi_flow_desag, unattributed_flow, aggregated_flow, 
                                                       offset = 2 if alg_name == "SSSD" else 0)
            prop_sp = flow_proportion_shortest_paths (multi_flow_desag, unattributed_flow, adj_mat, transport_times, pairs, 
                                                      offset = 2 if alg_name == "SSSD" else 0)
            #prop_fval_sp = flow_val_proportion_shortest_paths (multi_flow, adj_mat, transport_times, pairs, flow_values, 
            #                                                   offset = 2 if alg_name == "SSSD" else 0)
            print("Case ", update_transport_time, path_type, mode)
            print("Proportion of support arcs  ", prop_fsupp)
            print("Proportion of flow val residue ", flow_val_res)
            print("Proportion of flow residue  ", flow_res)
            print("Proportion of multi flow residue  ", m_flow_res)
            #time.sleep(0.5)
            if alg_name == "SSSD":
                dict_result[(i, update_transport_time, path_type, mode)] = (flow_val_res, flow_res, m_flow_res, prop_fsupp, prop_sp)
            
            elif alg_name == "MSMD":
                dict_result[(i, update_transport_time, path_type, mode, pair_criteria)] = (flow_val_res, flow_res, m_flow_res, prop_fsupp, prop_sp)

    return dict_result



######################################################################################################################################
#######################################   This function evaluates the real instances of the restore-problem (lieu-saint)   ###########
######################################################################################################################################
def eval_real_instances(dir_name_graph_instance, dir_name_multi_flow_instance, nb_instances,
                       update_transport_time = True, path_type = "random", mode = "capacity", pair_criteria = None,
                       maximal_flow_amount = float("inf")):
    # File names of the instances
    dict_result = {}
    for i in range(nb_instances):
        # Read the graph
        print("-------------------------------------------------------------------------------------")
        print("Treating instance ", i)
        # Read the (complete) instance
        if pair_criteria is None:
            complete_instance = read_ntreat_instance (dir_name_graph_instance = dir_name_graph_instance, 
                                                    dir_name_multi_flow_instance = dir_name_multi_flow_instance, 
                                                    id_instance = i,
                                                    instance_type = "real")
        else:
            complete_instance = read_ntreat_instance (dir_name_graph_instance = dir_name_graph_instance, 
                                                    dir_name_multi_flow_instance = dir_name_multi_flow_instance, 
                                                    id_instance = i,
                                                    instance_type = "real", 
                                                    add_supsource_supdestination = False)
        # Unpack the instance           
        adj_mat = complete_instance["adj_mat"]
        aggregated_flow = complete_instance["aggregated_flow"]
        transport_times = complete_instance["transport_times"]
        pairs = complete_instance["pairs"]
        flow_values = complete_instance["flow_values"]
        aggregated_flow = complete_instance["aggregated_flow"]
        multi_flow = complete_instance["multi_flow"]

        if pair_criteria is None:
            # Create an instance of 'MultiFLowDesagSolver'
            mf_decomp = MultiFlowDesagSolverSSSD(adj_mat = adj_mat, 
                                                aggregated_flow = aggregated_flow,
                                                transport_times = transport_times, 
                                                pairs = pairs,
                                                flow_values = flow_values,
                                                update_transport_time = update_transport_time,
                                                maximal_flow_amount = maximal_flow_amount)
        else:
            # Create an instance of 'MultiFLowDesagSolver'
            mf_decomp = MultiFlowDesagSolverMSMD(adj_mat = adj_mat, 
                                                aggregated_flow = aggregated_flow,
                                                transport_times = transport_times, 
                                                pairs = pairs,
                                                flow_values = flow_values,
                                                update_transport_time = update_transport_time,
                                                maximal_flow_amount = maximal_flow_amount)
            
        #  Use to desaggregate the multiflow with a call to method 'heuristic_multi_flow_desagregation'
        multi_flow_desag, flow_vals_desagg = mf_decomp.heuristic_multi_flow_desagregation (path_type = path_type,
                                                                                           mode = mode,
                                                                                           pair_criteria = pair_criteria)
        unattributed_flow = mf_decomp.unattributed_flow

        # Process the metrics and store them in 'dict_result'
        flow_val_res = flow_val_residue (flow_vals_desagg, flow_values)
        flow_res = flow_residue (multi_flow_desag, unattributed_flow, aggregated_flow, 
                                 offset = 2 if pair_criteria is None else 0)
        m_flow_res = multi_flow_residue (multi_flow_desag, multi_flow, aggregated_flow, 
                                         offset = 2 if pair_criteria is None else 0)
        prop_fsupp = proportion_size_flow_support (multi_flow_desag, unattributed_flow, aggregated_flow, 
                                                   offset = 2 if pair_criteria is None else 0)
        prop_sp = flow_proportion_shortest_paths (multi_flow_desag, unattributed_flow, adj_mat, transport_times, pairs, 
                                                  offset = 2 if pair_criteria is None else 0)
        #prop_fval_sp = flow_val_proportion_shortest_paths (multi_flow, adj_mat, transport_times, pairs, flow_values, offset = 0)
        print("Case ", update_transport_time, path_type, mode)
        print("Proportion of support arcs  ", prop_fsupp)
        print("Proportion of flow val residue ", flow_val_res)
        print("Proportion of flow residue  ", flow_res)
        print("Proportion of multi flow residue  ", m_flow_res)
        #time.sleep(0.5)
        dict_result[i] = (flow_val_res, flow_res, m_flow_res, prop_fsupp, prop_sp)

    return dict_result




######################
### Problème instance 18 de multi flow sur lieu saint !!!!!!!!!!!!!!!!!!!!!!
### !!!!!!!!!!!!!!!!!!!!!! à traiter
######################


def main():
    # Test names
    test_names = {"eval_real_instance", "eval_simulated_instances", 
                  "read_perfs_statistics", "display_difference_m_flow_residue",
                  "test_correlation", "compare_SSSD_MSMD",
                  "instance_lookup",
                  "comp_simulated_instances_pairs_criterias",
                  "comp_MSMD_vs_MSMD_Trans_Func",
                  "comp_MSMD_Trans_Func_vs_MSMD_Trans_Func_MulP"}

    test_name = "comp_MSMD_vs_MSMD_Trans_Func"

    # Algorithm names
    algorithm_names_informations = ["SSSD","MSMD", ["MSMD", "trans_func"]]
    alg_name_info = ["MSMD", "trans_func"]

    ######################
    ### Main file used is 'results.npy', it contains a dict whose structure is described as follows
    ### Pour SSSD:
    ### dict_result[(i, update_transport_time, path_type, mode)] = (flow_val_res, flow_res, m_flow_res, prop_fsupp, prop_sp, prop_fval_sp)
    ### Pour MSMD :
    ### dict_result[(i, update_transport_time, path_type, mode, pair_criteria)] = (flow_val_res, flow_res, m_flow_res, prop_fsupp, prop_sp, prop_fval_sp)
    ######################

    if test_name not in test_names or alg_name_info not in algorithm_names_informations:
        print("Test name or test algorithm unrecognized.")
        sys.exit()

    if test_name == "eval_simulated_instances": # Saved instances for comparing different modes and path selection strategies
        alg_name = alg_name_info if isinstance(alg_name_info, str) else alg_name_info[0]
        # To add cases to the tested cases if needed
        add_cases = False
        # Process the tested cases 
        if alg_name == "SSSD":
            test_cases = list(product([False, True], set(PATH_TYPES)-{"random_nfilter_byshortest_path"}, MODES, ["-"]))
            if add_cases:
                test_cases += [(False, "random_nfilter_byshortest_path", "max_capacity", "-"), 
                               (True, "random_nfilter_byshortest_path", "max_capacity", "-")]
        
        elif alg_name == "MSMD":
            pairs_criterias = {"max_remaining_flow_val"}
            the_mode2 = None
            test_cases = list(product([False, True], set(PATH_TYPES)-{"random_nfilter_byshortest_path"}, MODES, pairs_criterias))
            if add_cases:
                #test_cases += [(upd_t, "random_nfilter_byshortest_path", "max_capacity", p_crit) for p_crit in pairs_criterias
                #                                                                                    for upd_t in [True, False]]
                the_mode2 = "min_distance"
                test_cases += [(upd_t, "random_nfilter_byshortest_path", the_mode2, p_crit) for p_crit in pairs_criterias
                                                                                                    for upd_t in [True, False]]
        
        # Evaluate the performances of the instances 
        dict_result = eval_simulated_instances(dir_name_graph_instance = "instance_generation/instances/capacity/", 
                                               dir_name_multi_flow_instance = "multi_flow_generation/transition_function_instances", 
                                               nb_instances = 100,
                                               test_cases = test_cases,
                                               alg_name_info = alg_name_info,
                                               maximal_flow_amount = float("inf"),
                                               trans_func = True)
        # Save the performances
        save_performances_metrics (dict_result, 
                                   dir_save_name = "results/simulated/",
                                   alg_name_info = alg_name_info)
        # Draw the results in scatterplots
        if alg_name == "SSSD":
            display_perfs_simulated_instances_SSSD(dict_result = dict_result, 
                                                   dir_save_name = "results/simulated/"+alg_name+"/")
        elif alg_name == "MSMD":
            display_perfs_simulated_instances_MSMD(dict_result = dict_result, 
                                                   dir_save_name = "results/simulated/"+alg_name+"/",
                                                   the_mode2 = the_mode2)
    
    elif test_name == "comp_simulated_instances_pairs_criterias": # Only for comparing different criterias for selecting pairs
        alg_name = alg_name_info if isinstance(alg_name_info, str) else alg_name_info[0]
        #pairs_criterias = PAIRS_CRITERIAS
        pairs_criterias = {"max_remaining_flow_val"}
        #test_cases = list(product([True], {"random"}, {"min_distance", "max_capacity"}, pairs_criterias))
        test_cases = list(product([True], {"random"}, {"max_capacity"}, pairs_criterias))
        add_cases = True
        if add_cases:
            the_mode2 = "min_distance"
            test_cases += [(upd_t, "random_nfilter_byshortest_path", the_mode2, p_crit) for p_crit in pairs_criterias
                                                                                            for upd_t in [True]]
        #test_cases = [(True, "random", "min_distance", "max_remaining_flow_val")]
        nb_instances = 100
        maximal_flow_amount = 1
        dict_result = eval_simulated_instances(dir_name_graph_instance = "instance_generation/instances/capacity/", 
                                               dir_name_multi_flow_instance = "multi_flow_generation/instances/", 
                                               nb_instances = nb_instances,
                                               test_cases = test_cases,
                                               alg_name = alg_name,
                                               maximal_flow_amount = maximal_flow_amount)
        
        compare_ndispplay_pairs_criterias_perfs_simul_instances_MSMD (dict_result = dict_result, 
                                                                      dir_save_name = "results/simulated/MSMD/",
                                                                      update_transport_time = True, 
                                                                      path_types = ["random"], 
                                                                      modes = ["max_capacity"])
        print("Number of instances ", nb_instances)
        print("Total number of times flow is equal among pairs ", return_nb_equal_flow_val())
        print("Total number of iterations ", return_nb_total_it())
        print("Proportion of times flow is equal among pairs ", return_nb_equal_flow_val()/return_nb_total_it())

    elif test_name == "comp_MSMD_vs_MSMD_Trans_Func": # Only for comparing different criterias for selecting pairs
        alg_name = alg_name if isinstance(alg_name_info, str) else alg_name_info[0]
        compare_MSMD_vs_MSMD_Trans_Func (dir_name_graph_instance = "instance_generation/instances/capacity/",
                                         dir_name_multi_flow_instance1 = "multi_flow_generation/transition_function_instances/",
                                         dir_name_multi_flow_instance2 = "multi_flow_generation/transition_function_instances/", 
                                         dir_save_name = "results/simulated/MSMD/",
                                         nb_instances = 100,
                                         update_transport_time = True, 
                                         path_type = "random", 
                                         mode = "min_distance",
                                         pair_criteria = "max_remaining_flow_val",
                                         id_metric = 2,
                                         dict_titles = {0:"flow_val_res", 
                                                        1:"flow_res", 
                                                        2:"m_flow_res", 
                                                        3:"prop_fsupp", 
                                                        4:"prop_sp", 
                                                        5:"prop_fval_sp"},
                                         maximal_flow_amount = 1,
                                         legend_msmd = "msmd",
                                         legend_msmd_transf = "msmd_transf",
                                         nb_max_tries = 30,
                                         max_path_length = 10000,
                                         update_transition_functions = True,
                                         paths_use_only_trans_func = True,
                                         max_trans_func_successor = True,
                                         continue_best_effort = False)
    
    elif test_name == "comp_MSMD_Trans_Func_vs_MSMD_Trans_Func_MulP":
        alg_name = alg_name if isinstance(alg_name_info, str) else alg_name_info[0]
        compare_MSMD_Trans_Func_vs_MSMD_Trans_Func_MulP (
                                     dir_name_graph_instance = "instance_generation/instances/capacity/", 
                                     dir_name_multi_flow_instance = "multi_flow_generation/transition_function_instances/",
                                     dir_save_name = "results/simulated/MSMD/",
                                     nb_instances = 100,
                                     update_transport_time = True,
                                     pair_criteria = "max_remaining_flow_val",
                                     id_metric = 2,
                                     dict_titles = {0:"flow_val_res", 
                                                    1:"flow_res", 
                                                    2:"m_flow_res", 
                                                    3:"prop_fsupp", 
                                                    4:"prop_sp", 
                                                    5:"prop_fval_sp"},
                                     maximal_flow_amount = 1,
                                     legend_msmd_transf = "msmd_transf",
                                     legend_msmd_transf_mulp = "msmd_transf_mulp",
                                     nb_max_tries = 30,
                                     max_path_length = 10000,
                                     update_transition_functions = True,
                                     paths_use_only_trans_func = True,
                                     continue_best_effort = False,
                                     reodering_pairs_policy_name = "remaining_max_flow")

    elif test_name == "eval_real_instance":
        update_transport_time, path_type, mode, pair_criteria = True, "random", "max_capacity", "max_remaining_flow_val"
        dict_result = eval_real_instances(dir_name_graph_instance = "multi_flow_generation/real_instances/", 
                                            dir_name_multi_flow_instance = "multi_flow_generation/real_instances/capacity/", 
                                            nb_instances = 50,
                                            update_transport_time = update_transport_time,
                                            path_type = path_type, 
                                            mode = mode,
                                            pair_criteria = None if alg_name == "SSSD" else pair_criteria,
                                            maximal_flow_amount = float("inf"))
        dict_statistics = process_statistics_real (dict_result, mode)
        print(dict_statistics)
    
    elif test_name == "read_perfs_statistics":
        alg_name = alg_name if isinstance(alg_name_info, str) else alg_name_info[0]
        path_file_results = "results/simulated/results_SSSD.npy" if alg_name == "SSSD" else "results/simulated/results_MSMD.npy"
        pair_criteria = "max_remaining_flow_val"
        dict_results = np.load(path_file_results, allow_pickle = True).flat[0]
        # Process and save performances on statistics
        process_statistics(dict_results, 
                           update_transport_time = True, 
                           path_type = "random", 
                           dir_save_name = "results/simulated/",
                           chosen_pair_criteria = None if alg_name == "SSSD" else pair_criteria)
        
        path_file_statistics =  "results/simulated/perfs_SSSD.npy" if alg_name == "SSSD" else "results/simulated/perfs_MSMD.npy"
        perfs_statistics = np.load(path_file_statistics, allow_pickle = True).flat[0]
        pprint.pprint(perfs_statistics)

    elif test_name == "display_difference_m_flow_residue":
        alg_name = alg_name if isinstance(alg_name_info, str) else alg_name_info[0]
        path_file_results = "results/simulated/results_SSSD.npy" if alg_name == "SSSD" else "results/simulated/results_MSMD.npy"
        dict_results = np.load(path_file_results, allow_pickle = True).flat[0]
        if alg_name == "SSSD":
            display_multi_flow_residues_perfs_SSSD (dict_results,
                                                    dir_save_name = "results/simulated/SSSD/", 
                                                    chosen_update_transport_time = True,
                                                    chosen_path_type = "random")
        elif alg_name == "MSMD":
            display_multi_flow_residues_perfs_MSMD (dict_results, 
                                                    dir_save_name = "results/simulated/MSMD/", 
                                                    chosen_update_transport_time = True, 
                                                    chosen_path_type = "random",
                                                    chosen_pair_criteria = "max_remaining_flow_val")
        
    elif test_name == "test_correlation": ###########   !!!! Only for SSSD case for now !!!!   ###########
        alg_name = alg_name if isinstance(alg_name_info, str) else alg_name_info[0]
        path_file_results = "results/simulated/results_SSSD.npy" if alg_name == "SSSD" else "results/simulated/results_MSMD.npy"
        dict_results = np.load(path_file_results, allow_pickle = True).flat[0]
        test_correlation(dict_result = dict_results, 
                         update_transport_time = True, 
                         chosen_path_type = "random",
                         chosen_mode = "max_capacity")
        
    elif test_name == "compare_SSSD_MSMD":
        dict_results_sssd = np.load("results/simulated/results_SSSD.npy", allow_pickle = True).flat[0]
        dict_results_msmd = np.load("results/simulated/results_MSMD.npy", allow_pickle = True).flat[0]
        display_comparison_SSSD_MSMD (dir_save_name = "results/simulated/",
                                      dict_results_sssd = dict_results_sssd, 
                                      dict_results_msmd = dict_results_msmd, 
                                      params_sssd = {"update_transport_time":True, 
                                                     "path_type":"random",
                                                     "mode":"max_capacity"}, 
                                      params_msmd = {"update_transport_time":True, 
                                                     "path_type":"random",
                                                     "mode":"min_distance",
                                                     "pair_criteria":"max_remaining_flow_val"},
                                      legend_sssd = "SSSD", 
                                      legend_msmd = "MSMD")
    
    elif test_name == "instance_lookup":
        dict_results_msmd = np.load("results/simulated/results_MSMD.npy", allow_pickle = True).flat[0]
        ids_min = [key[0] for key, val in dict_results_msmd.items() if val[0] == 0 and val[1] == 0 and key[2] == "first_fit"]
        print(ids_min)
        for id in ids_min:
            print("ID ", id)
            print("prop_fval_sp ", dict_results_msmd[(id, True, "first_fit", "min_distance", "max_remaining_flow_val")][-1])
            print("prop_sp ", dict_results_msmd[(id, True, "first_fit", "min_distance", "max_remaining_flow_val")][-2])
        max_metric = max(val[1] for val in dict_results_msmd.values())
        for key, val in dict_results_msmd.items():
            if val[1] == max_metric:
                max_key = key
                break
        print("ID ", max_key[0])
        print("prop_fval_sp ", dict_results_msmd[max_key][-1])
        print("prop_sp ", dict_results_msmd[max_key][-2])



if __name__ == "__main__":
    main()