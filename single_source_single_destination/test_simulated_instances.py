from itertools import product

import os

import sys

sys.path.append(os.getcwd())
from multi_flow_desagregation_heuristics import MultiFlowDesagSolverSSSD

from utils.pre_process_utils import read_ntreat_instance

from instance_generation.random_instance_generator import read_data

from instance_generation.display_utils_gen import display_instance

from instance_generation.connexity import is_connected

from utils.metrics import flow_val_residue, flow_residue, multi_flow_residue, proportion_size_flow_support


def main():
    id_instance = 10
    show_graph = True
    if show_graph:
        file_path = "instance_generation/instances/capacity/instance_"+str(id_instance)+".npy"
        instance = read_data(file_path, read_all = True)
        print(instance["pairs"])
        display_instance (adj_matrice = instance["adj_mat"], 
                            arcs = instance["arcs"],
                            nodes = instance["nodes"], 
                            capacities = instance["capacities"], 
                            transport_times = instance["transport_times"],
                            pairs = instance["pairs"],
                            weight_pairs = instance["weight_pairs"],
                            print_ = False)
        sys.exit()
    
    complete_instance = read_ntreat_instance (dir_name_graph_instance = "instance_generation/instances/capacity/", 
                                                    dir_name_multi_flow_instance = "multi_flow_generation/instances/", 
                                                    id_instance = id_instance)
    # Unpack the instance
    adj_mat = complete_instance["adj_mat"]
    transport_times = complete_instance["transport_times"]
    pairs = complete_instance["pairs"]
    flow_values = complete_instance["flow_values"]
    aggregated_flow = complete_instance["aggregated_flow"]
    multi_flow = complete_instance["multi_flow"]
    
    # Desaggregate the flow for each cas in 'product' case and return the result in dict 'dict_result'
    update_transport_time, path_type, mode = False, "first_fit", "min_distance"
    mf_decomp = MultiFlowDesagSolverSSSD(adj_mat = adj_mat, 
                                            aggregated_flow = aggregated_flow,
                                            transport_times = transport_times, 
                                            pairs = pairs,
                                            flow_values = flow_values,
                                            update_transport_time = update_transport_time)
    #print("!!! IS the graph CONNECTED !!!", is_connected(adj_mat, source_node = len(adj_mat) - 2))
    multi_flow_desag, flow_vals_desagg = mf_decomp.heuristic_multi_flow_desagregation (path_type = path_type,
                                                                                       mode = mode)
    unattributed_flow = mf_decomp.unattributed_flow
    print("Pairs ", pairs)
    print("Flow values ", flow_values)

    sources = set(pair[0] for pair in pairs)
    sources_count = {s:len({i for i in range(len(pairs)) if pairs[i][0] == s}) for s in sources}
    destinations = set(pair[1] for pair in pairs)
    destinations_count = {d:len({i for i in range(len(pairs)) if pairs[i][1] == d}) for d in destinations}
    all_count = sources_count
    all_count.update({d_key:all_count.get(d_key, 0) + destinations_count[d_key] for d_key in destinations_count})
    print(all_count)

    flow_val_res = flow_val_residue (flow_vals_desagg, flow_values)
    flow_res = flow_residue (multi_flow_desag, unattributed_flow, aggregated_flow, offset = 2)
    m_flow_res = multi_flow_residue (multi_flow_desag, multi_flow, aggregated_flow, offset = 2)
    prop = proportion_size_flow_support (multi_flow_desag, unattributed_flow, aggregated_flow, offset = 2)

    print("Flow val residu ", flow_val_res)
    print("Flow residu ", flow_res)
    print("Multi Flow residu ", m_flow_res)
    print("Prop arcs flow ", prop)


if __name__ == "__main__":
    main()