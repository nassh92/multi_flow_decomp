from utils.general_eval_msmd_metrics import (transition_function_residue, 
                                             flow_val_residue, 
                                             flow_residue,
                                             multi_flow_residue,
                                             proportion_size_flow_support,
                                             flow_proportion_shortest_paths)
import sys


def process_key(ind_instance, test_infos, opt_params):
    if opt_params is None:
        return (ind_instance, test_infos)
    else:
        return (opt_params[0], opt_params[1], ind_instance, test_infos) 


def update_dict_results(dict_results, key, val):
    dict_results[key] = val


def process_performances(flow_vals_desagg, 
                         original_flow_values, 
                         multi_flow_desag,
                         original_aggregated_flow,
                         original_multi_flow,
                         graph,
                         ideal_transport_times,
                         pairs,
                         original_transition_function,
                         solver,
                         dict_results,
                         ind_instance,
                         test_infos,
                         graph_representation = "adjacency_matrix",
                         opt_params = None,
                         print_ = False):
    # Process the metrics and store them in 'dict_result'
    flow_val_res = flow_val_residue (flow_vals_desagg, 
                                    original_flow_values)
    flow_res = flow_residue (multi_flow_desag, 
                            original_aggregated_flow,
                            graph)
    m_flow_res = multi_flow_residue (multi_flow_desag, 
                                    original_multi_flow, 
                                    original_aggregated_flow,
                                    graph)
    prop_fsupp = proportion_size_flow_support (multi_flow_desag, 
                                               original_aggregated_flow, 
                                               graph)
    prop_sp = flow_proportion_shortest_paths (multi_flow_desag, 
                                              graph, 
                                              ideal_transport_times, 
                                              pairs,
                                              graph_representation = graph_representation)
    """if prop_sp == None and exit_on_none:
        print(multi_flow_desag)
        sys.exit()"""
    trans_func_res = transition_function_residue (original_transition_function, 
                                                solver.constructed_transition_function, 
                                                original_aggregated_flow,
                                                graph)
    
    coeff1, coeff2, coeff3 = test_infos[-1]
    total_weight_error = coeff1 * flow_val_res + coeff2 * flow_res + coeff3 * trans_func_res
    if total_weight_error > 1.0000001 or total_weight_error < 0:
        print("Total weigted error is out of range.")
        sys.exit()
    reward = 1 - total_weight_error

    if print_:
        print("Proportion of support arcs  ", prop_fsupp)
        print("Proportion of flow val residue ", flow_val_res)
        print("Proportion of flow residue  ", flow_res)
        print("Proportion of multi flow residue  ", m_flow_res)
        print("Proportion of trans func residue  ", trans_func_res)
        print("Original flow values ", original_flow_values)
        print("Desaggregated Flow values ", flow_vals_desagg)
    #time.sleep(0.5)
    key = process_key(ind_instance, test_infos, opt_params)
    update_dict_results(dict_results, 
                        key, 
                        (flow_val_res, flow_res, 
                        m_flow_res, prop_fsupp, 
                        prop_sp, trans_func_res,
                        reward))