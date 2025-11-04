import os

from itertools import product

import numpy as np

import time

import matplotlib.pyplot as plt

import sys

import statistics

from scipy.stats import pearsonr

import pprint

from copy import deepcopy

sys.path.append(os.getcwd())
from utils.metrics import (flow_val_residue, 
                     flow_residue, 
                     multi_flow_residue, 
                     proportion_size_flow_support, 
                     flow_proportion_shortest_paths, 
                     flow_val_proportion_shortest_paths)

from utils.pre_process_utils import read_ntreat_instance

from single_source_single_destination.multi_flow_desagregation_heuristics import (MultiFlowDesagSolverMSMD, 
                                                MultiFlowDesagSolverMSMDTransFunc,
                                                MultiFlowDesagSolverMSMDTransFuncMulP)

# Suppression of flow_val_proportion_shortest_paths

def save_performances_metrics (dict_result, dir_save_name, alg_name_info):
    alg_name = alg_name_info if isinstance(alg_name_info, str) else alg_name_info[0]
    # Save the performances of the heuristics
    if alg_name == "SSSD":
        np.save(dir_save_name+"results_SSSD", dict_result)
    
    elif alg_name == "MSMD":
        np.save(dir_save_name+"results_MSMD", dict_result)


######################################################################################################################################
#######################################   Functions for performing/displaying experiments with simulated instances   #################
######################################################################################################################################
def display_perfs_simulated_instances_SSSD(dict_result, dir_save_name):
    """
    Display data in raw form.
    """
    # Construct the temporary dict for display
    dict_temp = {}
    for (id_instance, update_transport_time, path_type, mode) in dict_result:
        if path_type != "random_nfilter_byshortest_path":
            elem = dict_result[(id_instance, update_transport_time, path_type, mode)]
            if (update_transport_time, path_type) not in dict_temp:
                dict_temp[(update_transport_time, path_type)] = [(id_instance, mode, elem[0], elem[1], elem[2], elem[3], elem[4], elem[5])]
            else: 
                dict_temp[(update_transport_time, path_type)].append((id_instance, mode, elem[0], elem[1], elem[2], elem[3], elem[4], elem[5]))
    
    print("-------------------------------------------------------------------------------------")
    # Sort every element of the newly created dict
    for key_dict in dict_temp: dict_temp[key_dict].sort(key = lambda x : (x[0], x[1]))

    # The goal here is to compare each type of 'path_type' : min_time and max_capacity
    for (update_transport_time, path_type) in dict_temp:
        modes = set(elem[1] for elem in dict_temp[(update_transport_time, path_type)])
        if "max_capacity" in modes:
            ### Draw for cas "max_capacity"
            mode = "max_capacity"
            print("--------------------------------", update_transport_time, path_type, mode, "--------------------------------")
            # Do the xs and ys
            xs_cap = [e[2] for e in dict_temp[(update_transport_time, path_type)] if e[1] == mode]
            ys_cap = [e[3] for e in dict_temp[(update_transport_time, path_type)] if e[1] == mode]
            proportions_cap = [e[5] for e in dict_temp[(update_transport_time, path_type)] if e[1] == mode]
            min_ = min(proportions_cap)
            max_ = max(proportions_cap)
            print("Min prop for case ", update_transport_time, path_type, mode, min_, np.argmin(proportions_cap))
            print("Max prop for case ", update_transport_time, path_type, mode, max_, np.argmax(proportions_cap))

            # Draw
            fig = plt.figure()
            title = "case_"+str(update_transport_time)+"_"+mode+"_"+path_type
            #plt.title(title)
            plt.scatter(xs_cap, ys_cap)

            # Save
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        if "min_distance" in modes:
            ### Draw for cas "min_distance"
            mode = "min_distance"
            print("--------------------------------", update_transport_time, path_type, mode, "--------------------------------")
            # Do the xs and ys
            xs_dist = [e[2] for e in dict_temp[(update_transport_time, path_type)] if e[1] == mode]
            ys_dist = [e[3] for e in dict_temp[(update_transport_time, path_type)] if e[1] == mode]
            proportions_dist = [e[5] for e in dict_temp[(update_transport_time, path_type)] if e[1] == mode]
            min_ = min(proportions_dist)
            max_ = max(proportions_dist)
            print("Min prop for case ", update_transport_time, path_type, mode, min_, np.argmin(proportions_dist))
            print("Max prop for case ", update_transport_time, path_type, mode, max_, np.argmax(proportions_dist))

            # Draw
            fig = plt.figure()
            title = "case_"+str(update_transport_time)+"_"+mode+"_"+path_type
            #plt.title(title)
            plt.scatter(xs_dist, ys_dist)

            # Save
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        
        if "max_capacity" in modes and "min_distance" in modes:
            ### Draw for both modes the curve for the comparaison flow_val_res vs flow_res
            fig = plt.figure()
            title = "case_flow_val_res_vs_flow_res_"+str(update_transport_time)+"_min_distance_vs_max_capacity_"+path_type
            #plt.title(title)
            plt.scatter(xs_cap, ys_cap, color="b", alpha=0.5, label="max_capacity")
            plt.scatter(xs_dist, ys_dist, color="r", alpha=0.5, label="min_time")
            plt.legend(loc="upper left")

            # Save
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Xs an Ys for the case 'max_capacity' and when we compare the flow_res vs m_flow_res
            xs_cap2 = [e[3] for e in dict_temp[(update_transport_time, path_type)] if e[1] == "max_capacity"]
            ys_cap2 = [e[4] for e in dict_temp[(update_transport_time, path_type)] if e[1] == "max_capacity"]
            
            # Xs an Ys for the case 'min_distance' and when we compare the flow_res vs m_flow_res
            xs_dist2 = [e[3] for e in dict_temp[(update_transport_time, path_type)] if e[1] == "min_distance"]
            ys_dist2 = [e[4] for e in dict_temp[(update_transport_time, path_type)] if e[1] == "min_distance"]


            fig = plt.figure()
            title = "case_flow_res_vs_m_flow_res_"+str(update_transport_time)+"_"+mode+"_"+path_type
            plt.scatter(xs_cap2, ys_cap2, color="b", alpha=0.5, label="max_capacity")
            plt.scatter(xs_dist2, ys_dist2, color="r", alpha=0.5, label="min_time")
            plt.legend(loc="upper left")
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Xs and Ys for the case whe we compare flow_res vs prop_sp
            xs_sp = [e[7] for e in dict_temp[(update_transport_time, path_type)] if e[1] == "min_distance"]
            ys_sp = [e[3] for e in dict_temp[(update_transport_time, path_type)] if e[1] == "min_distance"]
            fig = plt.figure()
            title = "case_flow_res_vs_pro_sp_"+str(update_transport_time)+"_min_distance_"+path_type
            plt.scatter(xs_sp, ys_sp, color="blue")
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Xs and Ys for the case whe we compare flow_res vs prop_sp
            xs_sp = [e[7] for e in dict_temp[(update_transport_time, path_type)] if e[1] == "max_capacity"]
            ys_sp = [e[3] for e in dict_temp[(update_transport_time, path_type)] if e[1] == "max_capacity"]
            fig = plt.figure()
            title = "case_flow_res_vs_pro_sp_"+str(update_transport_time)+"_max_capacity_"+path_type
            plt.scatter(xs_sp, ys_sp, color="blue")
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    # Comparing the case where we chose the shortest path from among those with maximum capacity
    # with to the other selection path policies
    # this corresponds to path_type == "random_nfilter_byshortest_path" vs others
    all_path_policies = set(key[-2] for key in dict_result)
    if "random_nfilter_byshortest_path" in all_path_policies:
        res1 = [(key[0],)+dict_result[key] for key in dict_result if key[1] == True and\
                                                                        key[2] == "random_nfilter_byshortest_path" and\
                                                                            key[3] == "max_capacity"]
        res1.sort(key = lambda x : x[0])
        res2 = [(key[0],)+dict_result[key] for key in dict_result if key[1] == True and\
                                                                        key[2] == "random" and\
                                                                            key[3] == "max_capacity"]
        res2.sort(key = lambda x : x[0])
        res3 = [(key[0],)+dict_result[key] for key in dict_result if key[1] == True and\
                                                                        key[2] == "first_fit" and\
                                                                            key[3] == "max_capacity"]
        res3.sort(key = lambda x : x[0])
        
        fig = plt.figure()
        title = "update_time/case_flow_val_res_vs_flow_res_True_max_capacity_random_nfilter_byshortest_path_"
        plt.scatter(x=[e[1] for e in res3], 
                    y=[e[2] for e in res3], 
                    color="b", alpha=0.5, label="max_capacity_first_fit")
        plt.scatter(x=[e[1] for e in res2], 
                    y=[e[2] for e in res2], 
                    color="g", alpha=0.5, label="max_capacity_random")
        plt.scatter(x=[e[1] for e in res1], 
                    y=[e[2] for e in res1], 
                    color="r", alpha=0.5, label="max_capacity_random_fil_by_ssps")
        plt.legend(loc="upper left")
        fig.savefig(dir_save_name+title, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        fig = plt.figure()
        title = "update_time/case_flow_res_vs_m_flow_res_True_max_capacity_random_nfilter_byshortest_path_"
        plt.scatter(x=[e[2] for e in res3], 
                    y=[e[3] for e in res3], 
                    color="b", alpha=0.5, label="max_capacity_first_fit")
        plt.scatter(x=[e[2] for e in res2], 
                    y=[e[3] for e in res2], 
                    color="g", alpha=0.5, label="max_capacity_random")
        plt.scatter(x=[e[2] for e in res1], 
                    y=[e[3] for e in res1], 
                    color="r", alpha=0.5, label="max_capacity_random_fil_by_ssps")
        plt.legend(loc="upper left")
        fig.savefig(dir_save_name+title, bbox_inches='tight', pad_inches=0)
        plt.close(fig)



def display_perfs_simulated_instances_MSMD(dict_result, 
                                           dir_save_name,
                                           the_mode2 = "max_capacity"):
    """
    Display data in raw form.
    """
    # Construct the temporary dict for display
    dict_temp = {}
    for (id_instance, update_transport_time, path_type, mode, pair_criteria) in dict_result:
        elem = dict_result[(id_instance, update_transport_time, path_type, mode, pair_criteria)]
        if (update_transport_time, path_type, pair_criteria) not in dict_temp:
            dict_temp[(update_transport_time, path_type, pair_criteria)] = [(id_instance, mode, elem[0], elem[1], elem[2], elem[3], elem[4], elem[5])]
        else: 
            dict_temp[(update_transport_time, path_type, pair_criteria)].append((id_instance, mode, elem[0], elem[1], elem[2], elem[3], elem[4], elem[5]))
    
    print("-------------------------------------------------------------------------------------")
    # Sort every element of the newly created dict
    for key_dict in dict_temp: dict_temp[key_dict].sort(key = lambda x : (x[0], x[1]))

    # The goal here is to compare each type of 'path_type' : min_time and max_capacity
    for (update_transport_time, path_type, pair_criteria) in dict_temp:
        modes = set(elem[1] for elem in dict_temp[(update_transport_time, path_type, pair_criteria)])
        if "max_capacity" in modes:
            ### Draw for cas "max_capacity"
            mode = "max_capacity"
            print("--------------------------------", update_transport_time, path_type, mode, pair_criteria,  "--------------------------------")
            # Do the xs and ys
            xs_cap = [e[2] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == mode]
            ys_cap = [e[3] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == mode]
            proportions_cap = [e[5] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == mode]
            min_ = min(proportions_cap)
            max_ = max(proportions_cap)
            print("Min prop for case ", update_transport_time, path_type, mode, pair_criteria, min_, np.argmin(proportions_cap))
            print("Max prop for case ", update_transport_time, path_type, mode, pair_criteria, max_, np.argmax(proportions_cap))

            # Draw
            fig = plt.figure()
            title = "case_"+str(update_transport_time)+"_"+mode+"_"+path_type+"_"+pair_criteria
            #plt.title(title)
            plt.scatter(xs_cap, ys_cap)

            # Save
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        if "min_distance" in modes:
            ### Draw for cas "min_distance"
            mode = "min_distance"
            print("--------------------------------", update_transport_time, path_type, mode, pair_criteria, "--------------------------------")
            # Do the xs and ys
            xs_dist = [e[2] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == mode]
            ys_dist = [e[3] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == mode]
            proportions_dist = [e[5] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == mode]
            min_ = min(proportions_dist)
            max_ = max(proportions_dist)
            print("Min prop for case ", update_transport_time, path_type, mode, pair_criteria, min_, np.argmin(proportions_dist))
            print("Max prop for case ", update_transport_time, path_type, mode, pair_criteria, max_, np.argmax(proportions_dist))

            # Draw
            fig = plt.figure()
            title = "case_"+str(update_transport_time)+"_"+mode+"_"+path_type+"_"+pair_criteria
            #plt.title(title)
            plt.scatter(xs_dist, ys_dist)

            # Save
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        
        if "max_capacity" in modes and "min_distance" in modes:
            ### Draw for both modes the curve for the comparaison flow_val_res vs flow_res
            fig = plt.figure()
            title = "case_flow_val_res_vs_flow_res_"+str(update_transport_time)+"_min_distance_vs_max_capacity_"+path_type+"_"+pair_criteria
            #plt.title(title)
            plt.scatter(xs_cap, ys_cap, color="b", alpha=0.5, label="max_capacity")
            plt.scatter(xs_dist, ys_dist, color="r", alpha=0.5, label="min_time")
            plt.legend(loc="upper left")

            # Save
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Xs an Ys for the case 'max_capacity' and when we compare the flow_res vs m_flow_res
            xs_cap2 = [e[3] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == "max_capacity"]
            ys_cap2 = [e[4] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == "max_capacity"]
            
            # Xs an Ys for the case 'min_distance' and when we compare the flow_res vs m_flow_res
            xs_dist2 = [e[3] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == "min_distance"]
            ys_dist2 = [e[4] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == "min_distance"]


            fig = plt.figure()
            title = "case_flow_res_vs_m_flow_res_"+str(update_transport_time)+"_"+mode+"_"+path_type+"_"+pair_criteria
            plt.scatter(xs_cap2, ys_cap2, color="b", alpha=0.5, label="max_capacity")
            plt.scatter(xs_dist2, ys_dist2, color="r", alpha=0.5, label="min_time")
            plt.legend(loc="upper left")
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Xs and Ys for the case whe we compare flow_res vs prop_sp
            xs_sp = [e[7] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == "min_distance"]
            ys_sp = [e[3] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == "min_distance"]
            fig = plt.figure()
            title = "case_flow_res_vs_pro_sp_"+str(update_transport_time)+"_min_distance_"+path_type+"_"+pair_criteria
            plt.scatter(xs_sp, ys_sp, color="blue")
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Xs and Ys for the case whe we compare flow_res vs prop_sp
            xs_sp = [e[7] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == "max_capacity"]
            ys_sp = [e[3] for e in dict_temp[(update_transport_time, path_type, pair_criteria)] if e[1] == "max_capacity"]
            fig = plt.figure()
            title = "case_flow_res_vs_pro_sp_"+str(update_transport_time)+"_max_capacity_"+path_type
            plt.scatter(xs_sp, ys_sp, color="blue")
            dir2 = "not_update_time/" if not update_transport_time else "update_time/"
            fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    # Comparing the case where we chose the shortest path from among those with 'maximum capacity'
    # with to the other selection path policies
    # this corresponds to path_type == "random_nfilter_byshortest_path" vs others
    all_path_policies = set(key[-3] for key in dict_result)
    if "random_nfilter_byshortest_path" in all_path_policies:
        res1 = [(key[0],)+dict_result[key] for key in dict_result if key[1] == True and\
                                                                        key[2] == "random_nfilter_byshortest_path" and\
                                                                            key[3] == the_mode2 and\
                                                                                key[4] == "max_remaining_flow_val"]
        res1.sort(key = lambda x : x[0])
        res2 = [(key[0],)+dict_result[key] for key in dict_result if key[1] == True and\
                                                                        key[2] == "random" and\
                                                                            key[3] == the_mode2 and\
                                                                                key[4] == "max_remaining_flow_val"]
        res2.sort(key = lambda x : x[0])
        res3 = [(key[0],)+dict_result[key] for key in dict_result if key[1] == True and\
                                                                        key[2] == "first_fit" and\
                                                                            key[3] == the_mode2 and\
                                                                                key[4] == "max_remaining_flow_val"]
        res3.sort(key = lambda x : x[0])
        
        fig = plt.figure()
        title = "update_time/case_flow_val_res_vs_flow_res_True_"+the_mode2+"_random_nfilter_byshortest_path_"
        plt.scatter(x=[e[1] for e in res3], 
                    y=[e[2] for e in res3], 
                    color="b", alpha=0.5, label=the_mode2+"_first_fit")
        plt.scatter(x=[e[1] for e in res2], 
                    y=[e[2] for e in res2], 
                    color="g", alpha=0.5, label=the_mode2+"_random")
        plt.scatter(x=[e[1] for e in res1], 
                    y=[e[2] for e in res1], 
                    color="r", alpha=0.5, label=the_mode2+"_random_fil_by_ssps")
        plt.legend(loc="upper left")
        fig.savefig(dir_save_name+title, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        fig = plt.figure()
        title = "update_time/case_flow_res_vs_m_flow_res_True_"+the_mode2+"_random_nfilter_byshortest_path_"
        plt.scatter(x=[e[2] for e in res3], 
                    y=[e[3] for e in res3], 
                    color="b", alpha=0.5, label=the_mode2+"_first_fit")
        plt.scatter(x=[e[2] for e in res2], 
                    y=[e[3] for e in res2], 
                    color="g", alpha=0.5, label=the_mode2+"_random")
        plt.scatter(x=[e[2] for e in res1], 
                    y=[e[3] for e in res1], 
                    color="r", alpha=0.5, label=the_mode2+"_random_fil_by_ssps")
        plt.legend(loc="upper left")
        fig.savefig(dir_save_name+title, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # dict_result[(i, update_transport_time, path_type, mode, pair_criteria)] = (flow_val_res, flow_res, m_flow_res, prop_fsupp, prop_sp, prop_fval_sp)
        res1.sort(key = lambda x : -x[3])
        res2.sort(key = lambda x : -x[3])
        res3.sort(key = lambda x : -x[3])
        fig = plt.figure()
        title = "update_time/case_sorted_m_flow_res_True_"+the_mode2+"_random_nfilter_byshortest_path"
        plt.scatter(x=list(range(len(res3))), 
                    y=[e[3] for e in res3], 
                    color="b", alpha=0.5, label=the_mode2+"_first_fit")
        plt.scatter(x=list(range(len(res2))), 
                    y=[e[3] for e in res2], 
                    color="g", alpha=0.5, label=the_mode2+"_random")
        plt.scatter(x=list(range(len(res1))), 
                    y=[e[3] for e in res1], 
                    color="r", alpha=0.5, label=the_mode2+"_random_fil_by_ssps")
        plt.legend(loc="upper left")
        fig.savefig(dir_save_name+title, bbox_inches='tight', pad_inches=0)
        plt.close(fig)



def compare_ndispplay_pairs_criterias_perfs_simul_instances_MSMD (dict_result, dir_save_name,
                                                                  update_transport_time, path_types, modes):
    path_type, mode = path_types[0], modes[0]
    # dict_result[(i, update_transport_time, path_type, mode, pair_criteria)] = (flow_val_res, flow_res, m_flow_res, prop_fsupp, prop_sp, prop_fval_sp)
    all_pair_criterias, colors = set(key[-1] for key in dict_result), ["c", "m", "y", "r", "b", "g"]
    it = 0
    fig = plt.figure()
    for pair_criteria in all_pair_criterias:
        res = [(key[0],)+dict_result[key] for key in dict_result if key[1] == update_transport_time and\
                                                                        key[2] == path_type and\
                                                                            key[3] == mode and\
                                                                                key[4] == pair_criteria]
        
        plt.scatter(x=[e[1] for e in res], 
                    y=[e[2] for e in res], 
                    color=colors[it], alpha=0.5, label=pair_criteria)
        it += 1
    dir2 = "not_update_time/" if not update_transport_time else "update_time/"
    title = "case_flow_val_res_vs_flow_res_comp_pair_criterias"
    plt.legend(loc="upper left")
    fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    it = 0
    fig1, fig2 = plt.figure(), plt.figure()
    for pair_criteria in all_pair_criterias:
        for path_type in path_types:
            for mode in modes:
                res = [(key[0],)+dict_result[key] for key in dict_result if key[1] == update_transport_time and\
                                                                                key[2] == path_type and\
                                                                                    key[3] == mode and\
                                                                                        key[4] == pair_criteria]
                
                res2 = [(key[0],)+dict_result[key] for key in dict_result if key[1] == update_transport_time and\
                                                                                key[2] == path_type and\
                                                                                    key[3] == mode and\
                                                                                        key[4] == pair_criteria]
                res2.sort(key = lambda x : -x[3])
                
                str_pair_criteria = pair_criteria[0]+"-"+pair_criteria[1] if isinstance(pair_criteria, tuple) else pair_criteria

                plt.figure(fig1.number)
                plt.scatter(x=[e[2] for e in res], 
                            y=[e[3] for e in res], 
                            color=colors[it], alpha=0.5, label=str_pair_criteria+" "+path_type+" "+mode)
                
                plt.figure(fig2.number)
                plt.scatter(x=list(range(len(res2))), 
                            y=[e[3] for e in res2], 
                            color=colors[it], alpha=0.5, label=str_pair_criteria+" "+path_type+" "+mode)
                it += 1
    
    plt.figure(fig1.number)
    dir2 = "not_update_time/" if not update_transport_time else "update_time/"
    title = "case_flow_res_vs_m_flow_res_comp_pair_criterias"
    plt.legend(loc="upper left")
    fig1.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)

    plt.figure(fig2.number)
    dir2 = "not_update_time/" if not update_transport_time else "update_time/"
    title = "case_sorted_m_flow_res_comp_pair_criterias"
    plt.legend(loc="upper left")
    fig2.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
    plt.close(fig2)
            



################################################################################################################################

def display_multi_flow_residues_perfs_SSSD (dict_result, dir_save_name, chosen_update_transport_time, chosen_path_type):
    # Construct the temporary dict for display
    dict_temp = {}
    for (id_instance, update_transport_time, path_type, mode) in dict_result:
        elem = dict_result[(id_instance, update_transport_time, path_type, mode)]
        if (update_transport_time, path_type, mode) not in dict_temp:
            dict_temp[(update_transport_time, path_type, mode)] = [(id_instance, elem[0], elem[1], elem[2], elem[3], elem[4], elem[5])]
        else: 
            dict_temp[(update_transport_time, path_type, mode)].append((id_instance, elem[0], elem[1], elem[2], elem[3], elem[4], elem[5]))

    for key_dict in dict_temp: dict_temp[key_dict].sort(key = lambda x : x[0])

    perfs_min_distance = [elem[3] for elem in dict_temp[(chosen_update_transport_time, chosen_path_type, "min_distance")]]
    perfs_max_capacity = [elem[3] for elem in dict_temp[(chosen_update_transport_time, chosen_path_type, "max_capacity")]]

    perfs_diff = [perfs_min_distance[i] - perfs_max_capacity[i] for i in range(len(perfs_max_capacity))]
    perfs_diff.sort(reverse = True)
    perfs_min_distance.sort(reverse=True)
    perfs_max_capacity.sort(reverse=True)

    fig = plt.figure()
    title = "scatter_absolute_m_flow_res_"+str(chosen_update_transport_time)+"_"+chosen_path_type
    plt.scatter(list(range(len(perfs_max_capacity))), perfs_max_capacity, color="b", alpha=0.5, label="max_capacity")
    plt.scatter(list(range(len(perfs_min_distance))), perfs_min_distance, color="r", alpha=0.5, label="min_time")
    plt.legend(loc="upper right")
    dir2 = "not_update_time/" if not chosen_update_transport_time else "update_time/"
    fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    fig = plt.figure()
    title = "bars_absolute_m_flow_res_"+str(chosen_update_transport_time)+"_"+chosen_path_type
    plt.bar(x=list(range(len(perfs_diff))), color="darkblue", height=perfs_diff)
    plt.legend(loc="upper right")
    dir2 = "not_update_time/" if not chosen_update_transport_time else "update_time/"
    fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def display_multi_flow_residues_perfs_MSMD (dict_result, dir_save_name, chosen_update_transport_time, 
                                            chosen_path_type, chosen_pair_criteria):
    # Construct the temporary dict for display
    dict_temp = {}
    for (id_instance, update_transport_time, path_type, mode, pair_criteria) in dict_result:
        elem = dict_result[(id_instance, update_transport_time, path_type, mode, pair_criteria)]
        if (update_transport_time, path_type, mode, pair_criteria) not in dict_temp:
            dict_temp[(update_transport_time, path_type, mode, pair_criteria)] = [(id_instance, elem[0], elem[1], elem[2], elem[3], elem[4], elem[5])]
        else: 
            dict_temp[(update_transport_time, path_type, mode, pair_criteria)].append((id_instance, elem[0], elem[1], elem[2], elem[3], elem[4], elem[5]))

    for key_dict in dict_temp: dict_temp[key_dict].sort(key = lambda x : x[0])

    perfs_min_distance = [elem[3] for elem in dict_temp[(chosen_update_transport_time, chosen_path_type, "min_distance", chosen_pair_criteria)]]
    perfs_max_capacity = [elem[3] for elem in dict_temp[(chosen_update_transport_time, chosen_path_type, "max_capacity", chosen_pair_criteria)]]

    perfs_diff = [perfs_min_distance[i] - perfs_max_capacity[i] for i in range(len(perfs_max_capacity))]
    perfs_diff.sort(reverse = True)
    perfs_min_distance.sort(reverse=True)
    perfs_max_capacity.sort(reverse=True)

    fig = plt.figure()
    title = "scatter_absolute_m_flow_res_"+str(chosen_update_transport_time)+"_"+chosen_path_type+"_"+chosen_pair_criteria
    plt.scatter(list(range(len(perfs_max_capacity))), perfs_max_capacity, color="b", alpha=0.5, label="max_capacity")
    plt.scatter(list(range(len(perfs_min_distance))), perfs_min_distance, color="r", alpha=0.5, label="min_time")
    plt.legend(loc="upper right")
    dir2 = "not_update_time/" if not chosen_update_transport_time else "update_time/"
    print(dir_save_name+dir2+title)
    fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    fig = plt.figure()
    title = "bars_absolute_m_flow_res_"+str(chosen_update_transport_time)+"_"+chosen_path_type+"_"+chosen_pair_criteria
    plt.bar(x=list(range(len(perfs_diff))), color="darkblue", height=perfs_diff)
    plt.legend(loc="upper right")
    dir2 = "not_update_time/" if not chosen_update_transport_time else "update_time/"
    fig.savefig(dir_save_name+dir2+title, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



##################################################################################################################################

def process_statistics(dict_result, update_transport_time, path_type, dir_save_name, chosen_pair_criteria = None):
    modes = set(key[-1] for key in dict_result) if chosen_pair_criteria is None else set(key[-2] for key in dict_result)
    dict_temp = {}
    if chosen_pair_criteria is None:
        for mode in modes:
            dict_temp[("flow_val_residue", mode)] = [dict_result[key][0] for key in dict_result if key[1] == update_transport_time and\
                                                                                                        key[2] == path_type and\
                                                                                                            key[3] == mode]

            dict_temp[("flow_residue", mode)] = [dict_result[key][1] for key in dict_result if key[1] == update_transport_time and\
                                                                                                        key[2] == path_type and\
                                                                                                            key[3] == mode]
            
            dict_temp[("multi_flow_residue", mode)] = [dict_result[key][2] for key in dict_result if key[1] == update_transport_time and\
                                                                                                        key[2] == path_type and\
                                                                                                            key[3] == mode]
    else:
        for mode in modes:
            dict_temp[("flow_val_residue", mode)] = [dict_result[key][0] for key in dict_result\
                                                                            if key[1] == update_transport_time and\
                                                                                key[2] == path_type and\
                                                                                    key[3] == mode and\
                                                                                        key[4] == chosen_pair_criteria]

            dict_temp[("flow_residue", mode)] = [dict_result[key][1] for key in dict_result if key[1] == update_transport_time and\
                                                                                                    key[2] == path_type and\
                                                                                                        key[3] == mode and\
                                                                                                            key[4] == chosen_pair_criteria]
            
            dict_temp[("multi_flow_residue", mode)] = [dict_result[key][2] for key in dict_result if key[1] == update_transport_time and\
                                                                                                        key[2] == path_type and\
                                                                                                            key[3] == mode and\
                                                                                                                key[4] == chosen_pair_criteria]
    
    dict_perfs = {}
    for key in dict_temp:
        dict_perfs[(key[0], key[1], "mean")] = round(statistics.mean(dict_temp[key]), 3)
        dict_perfs[(key[0], key[1], "std")] = round(statistics.stdev(dict_temp[key]), 3)
    
    if chosen_pair_criteria is None:
        np.save(dir_save_name+"perfs_SSSD", dict_perfs)
    else:
        np.save(dir_save_name+"perfs_MSMD", dict_perfs)


def test_correlation(dict_result, update_transport_time, chosen_path_type, chosen_mode):
    # Construct the temporary dict for display
    dict_temp = {}
    for (id_instance, update_transport_time, path_type, mode) in dict_result:
        elem = dict_result[(id_instance, update_transport_time, path_type, mode)]
        if (update_transport_time, path_type) not in dict_temp:
            dict_temp[(update_transport_time, path_type)] = [(id_instance, mode, elem[0], elem[1], elem[2], elem[3], elem[4], elem[5])]
        else: 
            dict_temp[(update_transport_time, path_type)].append((id_instance, mode, elem[0], elem[1], elem[2], elem[3], elem[4], elem[5]))

    for key_dict in dict_temp: dict_temp[key_dict].sort(key = lambda x : (x[0], x[1]))

    # Process the xs and the ys
    #xs_sp = [e[6] for e in dict_temp[(update_transport_time, path_type)]]
    xs_sp = [e[7] for e in dict_temp[(update_transport_time, chosen_path_type)] if e[1] == chosen_mode]
    ys_sp = [e[3] for e in dict_temp[(update_transport_time, chosen_path_type)] if e[1] == chosen_mode]

    # Process the Pearson Correlation Coefficient
    rho, p_val = pearsonr(xs_sp, ys_sp) 

    print("The value of the  Pearson Correlation Coefficient is ", round(rho, 3), " with p-value of ", round(p_val, 3))




################################################################################################################################
#######################################   Functions for doing experiments with real instances   ################################
################################################################################################################################
def process_statistics_real (dict_result, mode):
    dict_statistics, metrics = {}, ["flow_val_residue", "flow_residue", "multi_flow_residue"]
    for i in range(len(metrics)):
        dict_statistics[(metrics[i], mode, "mean")] = round(statistics.mean([elem[i] for _, elem in dict_result.items()]), 3)
        dict_statistics[(metrics[i], mode, "std")] = round(statistics.stdev([elem[i] for _, elem in dict_result.items()]), 3)
    
    return dict_statistics





######################################################################################################################################
################################################   Compare between SSSD and MSMD   ###################################################
######################################################################################################################################
def display_comparison_SSSD_MSMD (dir_save_name,
                                  dict_results_sssd, dict_results_msmd, 
                                  params_sssd, params_msmd,
                                  legend_sssd, legend_msmd):
    dict_perfs = {}

    # Construct the temporary dict for display sssd
    dict_perfs[("SSSD", "flow_val_residue")] = [(key[0], dict_results_sssd[key][0]) for key in dict_results_sssd\
                                                                                        if key[1] == params_sssd["update_transport_time"] and\
                                                                                            key[2] == params_sssd["path_type"] and\
                                                                                                key[3] == params_sssd["mode"]]
    dict_perfs[("SSSD", "flow_val_residue")].sort(key = lambda x : x[0]) 
    dict_perfs[("SSSD", "flow_residue")] = [(key[0], dict_results_sssd[key][1]) for key in dict_results_sssd\
                                                                                    if key[1] == params_sssd["update_transport_time"] and\
                                                                                        key[2] == params_sssd["path_type"] and\
                                                                                            key[3] == params_sssd["mode"]]
    dict_perfs[("SSSD", "flow_residue")].sort(key = lambda x : x[0])
    dict_perfs[("SSSD", "multi_flow_residue")] = [(key[0], dict_results_sssd[key][2]) for key in dict_results_sssd\
                                                                                        if key[1] == params_sssd["update_transport_time"] and\
                                                                                            key[2] == params_sssd["path_type"] and\
                                                                                                key[3] == params_sssd["mode"]]
    dict_perfs[("SSSD", "multi_flow_residue")].sort(key = lambda x : x[0])

    # Construct the temporary dict for display msmd
    dict_perfs[("MSMD", "flow_val_residue")] = [(key[0], dict_results_msmd[key][0]) for key in dict_results_msmd\
                                                                                        if key[1] == params_msmd["update_transport_time"] and\
                                                                                            key[2] == params_msmd["path_type"] and\
                                                                                                key[3] == params_msmd["mode"] and\
                                                                                                    key[4] == params_msmd["pair_criteria"]]
    dict_perfs[("MSMD", "flow_val_residue")].sort(key = lambda x : x[0]) 
    dict_perfs[("MSMD", "flow_residue")] = [(key[0], dict_results_msmd[key][1]) for key in dict_results_msmd\
                                                                                    if key[1] == params_msmd["update_transport_time"] and\
                                                                                        key[2] == params_msmd["path_type"] and\
                                                                                            key[3] == params_msmd["mode"] and\
                                                                                                key[4] == params_msmd["pair_criteria"]]
    dict_perfs[("MSMD", "flow_residue")].sort(key = lambda x : x[0])
    dict_perfs[("MSMD", "multi_flow_residue")] = [(key[0], dict_results_msmd[key][2]) for key in dict_results_msmd\
                                                                                        if key[1] == params_msmd["update_transport_time"] and\
                                                                                            key[2] == params_msmd["path_type"] and\
                                                                                                key[3] == params_msmd["mode"] and\
                                                                                                    key[4] == params_msmd["pair_criteria"]]
    dict_perfs[("MSMD", "multi_flow_residue")].sort(key = lambda x : x[0])

    # Renaming of the variables
    ss1, ss2, ss3 = params_sssd["update_transport_time"], params_sssd["path_type"], params_sssd["mode"]
    ms1, ms2, ms3, ms4 = params_msmd["update_transport_time"], params_msmd["path_type"], params_msmd["mode"], params_msmd["pair_criteria"]

    ### Draw for both modes the curve for the comparaison flow_val_res vs flow_res
    fig = plt.figure()
    title = "case_flow_val_res_vs_flow_res_"+str(ss1)+"_"+ss2+"_"+ss3+"_vs_"+str(ms1)+"_"+ms2+"_"+ms3+"_"+ms4
    x_sssd = [e[1] for e in dict_perfs[("SSSD", "flow_val_residue")]]
    y_sssd = [e[1] for e in dict_perfs[("SSSD", "flow_residue")]]
    x_msmd = [e[1] for e in dict_perfs[("MSMD", "flow_val_residue")]]
    y_msmd = [e[1] for e in dict_perfs[("MSMD", "flow_residue")]]
    #plt.title(title)
    plt.scatter(x_sssd, y_sssd, color="b", alpha=0.5, label=legend_sssd)
    plt.scatter(x_msmd, y_msmd, color="r", alpha=0.5, label=legend_msmd)
    plt.xlabel("flow_val_res")
    plt.ylabel("flow_res")
    plt.legend(loc="upper left")
    fig.savefig(dir_save_name+title, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig = plt.figure()
    title = "scatter_absolute_m_flow_res_"+str(ss1)+"_"+ss2+"_"+ss3+"_vs_"+str(ms1)+"_"+ms2+"_"+ms3+"_"+ms4
    y_sssd2 = [e[1] for e in dict_perfs[("SSSD", "multi_flow_residue")]]
    y_sssd2.sort(reverse=True)
    y_msmd2 = [e[1] for e in dict_perfs[("MSMD", "multi_flow_residue")]]
    y_msmd2.sort(reverse=True)
    plt.scatter(list(range(len(y_sssd2))), y_sssd2, color="b", alpha=0.5, label=legend_sssd)
    plt.scatter(list(range(len(y_msmd2))), y_msmd2, color="r", alpha=0.5, label=legend_msmd)
    plt.xlabel("Instances")
    plt.ylabel("multi_flow_res")
    plt.legend(loc="upper right")
    fig.savefig(dir_save_name+title, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



######################################################################################################################################
#####################################     Compare between MSMD and MSMD with transition function      ################################
######################################################################################################################################
def compare_MSMD_vs_MSMD_Trans_Func (dir_name_graph_instance, 
                                     dir_name_multi_flow_instance1,
                                     dir_name_multi_flow_instance2, 
                                     dir_save_name,
                                     nb_instances,
                                     update_transport_time, 
                                     path_type, 
                                     mode, 
                                     pair_criteria,
                                     id_metric,
                                     dict_titles,
                                     maximal_flow_amount = float("inf"),
                                     legend_msmd = "msmd",
                                     legend_msmd_transf = "msmd_transf",
                                     nb_max_tries = 1,
                                     max_path_length = 1000,
                                     update_transition_functions = False,
                                     paths_use_only_trans_func = False,
                                     max_trans_func_successor = False,
                                     continue_best_effort = False,):
    dict_result = {}
    # 
    for i in range(nb_instances):
        for trans_func in [False, True]:
            complete_instance = read_ntreat_instance (dir_name_graph_instance = dir_name_graph_instance, 
                                                      dir_name_multi_flow_instance = dir_name_multi_flow_instance1 if not trans_func else dir_name_multi_flow_instance2,
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
            if not trans_func:
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
            
            elif trans_func:
                # Create an instance of 'MultiFlowDesagSolverMSMDTransFunc'
                mf_decomp = MultiFlowDesagSolverMSMDTransFunc(adj_mat = adj_mat, 
                                                              aggregated_flow = aggregated_flow, 
                                                              transport_times = transport_times, 
                                                              pairs = pairs,
                                                              flow_values = flow_values, 
                                                              ls_transition_function = complete_instance["transition_functions"],
                                                              update_transport_time = update_transport_time,
                                                              update_transition_functions = update_transition_functions,
                                                              nb_max_tries = nb_max_tries,
                                                              max_path_length = max_path_length, 
                                                              maximal_flow_amount = maximal_flow_amount,
                                                              paths_use_only_trans_func = paths_use_only_trans_func,
                                                              max_trans_func_successor = max_trans_func_successor)
                #  Use to desaggregate the multiflow with a call to method 'heuristic_multi_flow_desagregation'
                multi_flow_desag, flow_vals_desagg = mf_decomp.heuristic_multi_flow_desagregation (pair_criteria = pair_criteria,
                                                                                                   continue_best_effort = continue_best_effort)
                unattributed_flow = mf_decomp.unattributed_flow
                
            # Process the metrics and store them in 'dict_result'
            flow_val_res = flow_val_residue (flow_vals_desagg, flow_values)
            flow_res = flow_residue (multi_flow_desag, unattributed_flow, aggregated_flow, 
                                     offset = 0)
            m_flow_res = multi_flow_residue (multi_flow_desag, multi_flow, aggregated_flow, 
                                             offset = 0)
            prop_fsupp = proportion_size_flow_support (multi_flow_desag, unattributed_flow, aggregated_flow, 
                                                       offset = 0)
            prop_sp = flow_proportion_shortest_paths (multi_flow_desag, unattributed_flow, adj_mat, transport_times, pairs, 
                                                      offset = 0)
            #prop_fval_sp = flow_val_proportion_shortest_paths (multi_flow, adj_mat, transport_times, pairs, flow_values, 
            #                                                   offset = 0)
            print("Case ", update_transport_time, path_type, mode)
            print("Proportion of support arcs  ", prop_fsupp)
            print("Proportion of flow val residue ", flow_val_res)
            print("Proportion of flow residue  ", flow_res)
            print("Proportion of multi flow residue  ", m_flow_res)
            #time.sleep(0.5)
            dict_result[(i, trans_func)] = (flow_val_res, flow_res, m_flow_res, prop_fsupp, prop_sp)


    fig = plt.figure()
    title = "scatter_"+dict_titles[id_metric]+"_"+str(update_transport_time)+"_pt_"+path_type+"_m_"+mode
    y_msmd = [dict_result[key][id_metric] for key in dict_result if key[1] == False]
    y_msmd.sort(reverse=True)
    y_msmd_transf = [dict_result[key][id_metric] for key in dict_result if key[1] == True]
    y_msmd_transf.sort(reverse=True)

    plt.scatter(list(range(len(y_msmd))), y_msmd, color="b", alpha=0.5, label=legend_msmd)
    plt.scatter(list(range(len(y_msmd_transf))), y_msmd_transf, color="r", alpha=0.5, label=legend_msmd_transf)
    plt.xlabel("Instances")
    plt.ylabel(dict_titles[id_metric])
    plt.legend(loc="upper right")
    fig.savefig(os.path.join(dir_save_name, title), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig = plt.figure()
    title = "scatter_"+dict_titles[id_metric]+"_"+dict_titles[0]+"_"+str(update_transport_time)+"_pt_"+path_type+"_m_"+mode
    x_msmd = [(key[0], dict_result[key][0]) for key in dict_result if key[1] == False]
    x_msmd.sort(key = lambda x : x[0])
    x_msmd = [e[1] for e in x_msmd]
    y_msmd = [(key[0], dict_result[key][id_metric]) for key in dict_result if key[1] == False]
    y_msmd.sort(key = lambda x : x[0])
    y_msmd = [e[1] for e in y_msmd]

    x_msmd_transf = [(key[0], dict_result[key][0]) for key in dict_result if key[1] == True]
    x_msmd_transf.sort(key = lambda x : x[0])
    x_msmd_transf = [e[1] for e in x_msmd_transf]
    y_msmd_transf = [(key[0], dict_result[key][id_metric]) for key in dict_result if key[1] == True]
    y_msmd_transf.sort(key = lambda x : x[0])
    y_msmd_transf = [e[1] for e in y_msmd_transf]

    plt.scatter(x_msmd, y_msmd, color="b", alpha=0.5, label=legend_msmd)
    plt.scatter(x_msmd_transf, y_msmd_transf, color="r", alpha=0.5, label=legend_msmd_transf)
    plt.xlabel(dict_titles[0])
    plt.ylabel(dict_titles[id_metric])
    plt.legend(loc="upper right")
    fig.savefig(os.path.join(dir_save_name, title), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


    fig = plt.figure()
    title = "scatter_"+dict_titles[id_metric]+"_"+dict_titles[1]+"_"+str(update_transport_time)+"_pt_"+path_type+"_m_"+mode
    x_msmd = [(key[0], dict_result[key][1]) for key in dict_result if key[1] == False]
    x_msmd.sort(key = lambda x : x[0])
    x_msmd = [e[1] for e in x_msmd]
    y_msmd = [(key[0], dict_result[key][id_metric]) for key in dict_result if key[1] == False]
    y_msmd.sort(key = lambda x : x[0])
    y_msmd = [e[1] for e in y_msmd]

    x_msmd_transf = [(key[0], dict_result[key][1]) for key in dict_result if key[1] == True]
    x_msmd_transf.sort(key = lambda x : x[0])
    x_msmd_transf = [e[1] for e in x_msmd_transf]
    y_msmd_transf = [(key[0], dict_result[key][id_metric]) for key in dict_result if key[1] == True]
    y_msmd_transf.sort(key = lambda x : x[0])
    y_msmd_transf = [e[1] for e in y_msmd_transf]

    plt.scatter(x_msmd, y_msmd, color="b", alpha=0.5, label=legend_msmd)
    plt.scatter(x_msmd_transf, y_msmd_transf, color="r", alpha=0.5, label=legend_msmd_transf)
    plt.xlabel(dict_titles[1])
    plt.ylabel(dict_titles[id_metric])
    plt.legend(loc="upper right")
    fig.savefig(os.path.join(dir_save_name, title), bbox_inches='tight', pad_inches=0)
    plt.close(fig)





######################################################################################################################################
#####################################     Compare between MSMD with transition function      #########################################
#####################################     and MSMD with transition function and choosing multiple paths  #############################
######################################################################################################################################
def compare_MSMD_Trans_Func_vs_MSMD_Trans_Func_MulP (
                                     dir_name_graph_instance, 
                                     dir_name_multi_flow_instance,
                                     dir_save_name,
                                     nb_instances,
                                     update_transport_time,
                                     pair_criteria,
                                     id_metric,
                                     dict_titles,
                                     maximal_flow_amount = float("inf"),
                                     legend_msmd_transf = "msmd",
                                     legend_msmd_transf_mulp = "msmd_transf",
                                     nb_max_tries = 1,
                                     max_path_length = 1000,
                                     update_transition_functions = False,
                                     paths_use_only_trans_func = False,
                                     continue_best_effort = False,
                                     reodering_pairs_policy_name = "uniform"):
    dict_result = {}
    for i in range(nb_instances):
        print("Instance ", i)
        complete_instance = read_ntreat_instance (dir_name_graph_instance = dir_name_graph_instance, 
                                                  dir_name_multi_flow_instance = dir_name_multi_flow_instance,
                                                  id_instance = i,
                                                  add_supsource_supdestination = False,
                                                  trans_func = True)
        adj_mat = complete_instance["adj_mat"]
        aggregated_flow = complete_instance["aggregated_flow"]
        transport_times = complete_instance["transport_times"]
        pairs = complete_instance["pairs"]
        flow_values = complete_instance["flow_values"]
        aggregated_flow = complete_instance["aggregated_flow"]
        multi_flow = complete_instance["multi_flow"]
        
        for multiple_paths in [False, True]:
            # Unpack the instance
            # Desaggregate the flow for each cas in 'test_cases' case and return the result in dict 'dict_result'
            if not multiple_paths:
                # Create an instance of 'MultiFlowDesagSolverMSMD'
                mf_decomp = MultiFlowDesagSolverMSMDTransFunc(adj_mat = deepcopy(adj_mat), 
                                                              aggregated_flow = deepcopy(aggregated_flow), 
                                                              transport_times = deepcopy(transport_times), 
                                                              pairs = deepcopy(pairs),
                                                              flow_values = deepcopy(flow_values), 
                                                              ls_transition_function = deepcopy(complete_instance["transition_functions"]),
                                                              update_transport_time = update_transport_time,
                                                              update_transition_functions = update_transition_functions,
                                                              nb_max_tries = nb_max_tries,
                                                              max_path_length = max_path_length, 
                                                              maximal_flow_amount = maximal_flow_amount,
                                                              paths_use_only_trans_func = paths_use_only_trans_func)
                #  Use to desaggregate the multiflow with a call to method 'heuristic_multi_flow_desagregation'
                multi_flow_desag, flow_vals_desagg = mf_decomp.heuristic_multi_flow_desagregation (pair_criteria = pair_criteria,
                                                                                                   continue_best_effort = continue_best_effort)
                unattributed_flow = mf_decomp.unattributed_flow
            
            elif multiple_paths:
                # Create an instance of 'MultiFlowDesagSolverMSMDTransFunc'
                mf_decomp = MultiFlowDesagSolverMSMDTransFuncMulP(
                                                                adj_mat = deepcopy(adj_mat), 
                                                                aggregated_flow = deepcopy(aggregated_flow), 
                                                                transport_times = deepcopy(transport_times), 
                                                                pairs = deepcopy(pairs), 
                                                                flow_values = deepcopy(flow_values), 
                                                                ls_transition_function = deepcopy(complete_instance["transition_functions"]),
                                                                update_transport_time = update_transport_time,
                                                                update_transition_functions = update_transition_functions,
                                                                paths_use_only_trans_func = paths_use_only_trans_func,
                                                                max_path_length = max_path_length, 
                                                                reodering_pairs_policy_name = reodering_pairs_policy_name
                                                              )
                #  Use to desaggregate the multiflow with a call to method 'heuristic_multi_flow_desagregation'
                multi_flow_desag, flow_vals_desagg = mf_decomp.heuristic_multi_flow_desagregation ()
                unattributed_flow = mf_decomp.unattributed_flow
            
            # Process the metrics and store them in 'dict_result'
            flow_val_res = flow_val_residue (flow_vals_desagg, flow_values)
            flow_res = flow_residue (multi_flow_desag, unattributed_flow, aggregated_flow, 
                                     offset = 0)
            m_flow_res = multi_flow_residue (multi_flow_desag, multi_flow, aggregated_flow, 
                                             offset = 0)
            prop_fsupp = proportion_size_flow_support (multi_flow_desag, unattributed_flow, aggregated_flow, 
                                                       offset = 0)
            prop_sp = flow_proportion_shortest_paths (multi_flow_desag, unattributed_flow, adj_mat, transport_times, pairs, 
                                                      offset = 0)
            #prop_fval_sp = flow_val_proportion_shortest_paths (multi_flow, adj_mat, transport_times, pairs, flow_values, 
            #                                                   offset = 0)
            print("Case ", update_transport_time)
            print("Proportion of support arcs  ", prop_fsupp)
            print("Proportion of flow val residue ", flow_val_res)
            print("Proportion of flow residue  ", flow_res)
            print("Proportion of multi flow residue  ", m_flow_res)
            #time.sleep(0.5)
            dict_result[(i, multiple_paths)] = (flow_val_res, flow_res, m_flow_res, prop_fsupp, prop_sp)


    fig = plt.figure()
    title = "scatter_"+dict_titles[id_metric]+"_"+str(update_transport_time)
    y_msmd_transf = [dict_result[key][id_metric] for key in dict_result if key[1] == False]
    y_msmd_transf.sort(reverse=True)
    y_msmd_transf_mulp = [dict_result[key][id_metric] for key in dict_result if key[1] == True]
    y_msmd_transf_mulp.sort(reverse=True)

    plt.scatter(list(range(len(y_msmd_transf))), y_msmd_transf, color="b", alpha=0.5, label=legend_msmd_transf)
    plt.scatter(list(range(len(y_msmd_transf_mulp))), y_msmd_transf_mulp, color="r", alpha=0.5, label=legend_msmd_transf_mulp)
    plt.xlabel("Instances")
    plt.ylabel(dict_titles[id_metric])
    plt.legend(loc="upper right")
    fig.savefig(os.path.join(dir_save_name, title), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig = plt.figure()
    title = "scatter_"+dict_titles[id_metric]+"_"+dict_titles[0]+"_"+str(update_transport_time)
    x_msmd_transf = [(key[0], dict_result[key][0]) for key in dict_result if key[1] == False]
    x_msmd_transf.sort(key = lambda x : x[0])
    x_msmd_transf = [e[1] for e in x_msmd_transf]
    y_msmd_transf = [(key[0], dict_result[key][id_metric]) for key in dict_result if key[1] == False]
    y_msmd_transf.sort(key = lambda x : x[0])
    y_msmd_transf = [e[1] for e in y_msmd_transf]

    x_msmd_transf_mulp = [(key[0], dict_result[key][0]) for key in dict_result if key[1] == True]
    x_msmd_transf_mulp.sort(key = lambda x : x[0])
    x_msmd_transf_mulp = [e[1] for e in x_msmd_transf_mulp]
    y_msmd_transf_mulp = [(key[0], dict_result[key][id_metric]) for key in dict_result if key[1] == True]
    y_msmd_transf_mulp.sort(key = lambda x : x[0])
    y_msmd_transf_mulp = [e[1] for e in y_msmd_transf_mulp]

    plt.scatter(x_msmd_transf, y_msmd_transf, color="b", alpha=0.5, label=legend_msmd_transf)
    plt.scatter(x_msmd_transf_mulp, y_msmd_transf_mulp, color="r", alpha=0.5, label=legend_msmd_transf_mulp)
    plt.xlabel(dict_titles[0])
    plt.ylabel(dict_titles[id_metric])
    plt.legend(loc="upper right")
    fig.savefig(os.path.join(dir_save_name, title), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


    fig = plt.figure()
    title = "scatter_"+dict_titles[id_metric]+"_"+dict_titles[1]+"_"+str(update_transport_time)
    x_msmd_transf = [(key[0], dict_result[key][1]) for key in dict_result if key[1] == False]
    x_msmd_transf.sort(key = lambda x : x[0])
    x_msmd_transf = [e[1] for e in x_msmd_transf]
    y_msmd_transf = [(key[0], dict_result[key][id_metric]) for key in dict_result if key[1] == False]
    y_msmd_transf.sort(key = lambda x : x[0])
    y_msmd_transf = [e[1] for e in y_msmd_transf]

    x_msmd_transf_mulp = [(key[0], dict_result[key][1]) for key in dict_result if key[1] == True]
    x_msmd_transf_mulp.sort(key = lambda x : x[0])
    x_msmd_transf_mulp = [e[1] for e in x_msmd_transf_mulp]
    y_msmd_transf_mulp = [(key[0], dict_result[key][id_metric]) for key in dict_result if key[1] == True]
    y_msmd_transf_mulp.sort(key = lambda x : x[0])
    y_msmd_transf_mulp = [e[1] for e in y_msmd_transf_mulp]

    plt.scatter(x_msmd_transf, y_msmd_transf, color="b", alpha=0.5, label=legend_msmd_transf)
    plt.scatter(x_msmd_transf_mulp, y_msmd_transf_mulp, color="r", alpha=0.5, label=legend_msmd_transf_mulp)
    plt.xlabel(dict_titles[1])
    plt.ylabel(dict_titles[id_metric])
    plt.legend(loc="upper right")
    fig.savefig(os.path.join(dir_save_name, title), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

