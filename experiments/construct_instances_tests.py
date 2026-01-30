import os

import sys

import numpy as np

sys.path.append(os.getcwd())
from msmd.multi_flow_desag_instance_utils import construct_instances


"""ls_vals = [(55, 10), (55, 15), (55, 20),
        (75, 10), (75, 20),
        (95, 10), (95, 15), (95, 20)]"""

ls_vals = [(95, 20)]

for nb_nodes, pairs in ls_vals:
    print("--------------------------------------------------------------")
    print(nb_nodes, pairs)
    print("--------------------------------------------------------------")
    #dir_name_graph_instance = "instance_generation/instances/capacity/"
    dir_name_graph_instance = "data/simulated_data/graph_instances/random/"
    dir_name_graph_instance += "instances_nbnodes="+str(nb_nodes)+"_pairs="+str(pairs)+"/"
    #dir_name_multi_flow_instance = "multi_flow_generation/transition_function_instances/"
    dir_name_multi_flow_instance = "data/simulated_data/complete_instances/multi_flow_instances/random/"
    dir_name_multi_flow_instance += "nb_nodes="+str(nb_nodes)+"_pairs="+str(pairs)+"/"
    dict_instances = construct_instances (
                                dir_name_graph_instance = dir_name_graph_instance, 
                                dir_name_multi_flow_instance = dir_name_multi_flow_instance,
                                nb_instances = 100,
                                ls_update_transport_time = [True],
                                ls_update_transition_functions = [True])
    np.save("data/simulated_data/complete_instances/node_pairs/data_instances_random_"+str(nb_nodes)+"_"+str(pairs), 
            dict_instances)