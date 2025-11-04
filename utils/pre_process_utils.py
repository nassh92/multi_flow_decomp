import numpy as np
import os
import sys
from instance_generation.random_instance_generator import read_data
from instance_generation.real_instance.saint_lieu_instance_generator import generate_instance_saint_lieu

def process_instance (adj_mat, transport_times, aggregated_flow, pairs, flow_values):
    # pre-processing for later. Calculate sources and sources values
    sources = set(pair[0] for pair in pairs)
    sources_indices = {s:{i for i in range(len(pairs)) if pairs[i][0] == s} for s in sources}
    sources_values = {s:sum(flow_values[i] for i in sources_indices[s]) for s in sources}

    # pre-processing for later. Calculate destinations and destinations values
    destinations = set(pair[1] for pair in pairs)
    destinations_indices = {d:{i for i in range(len(pairs)) if pairs[i][1] == d} for d in destinations}
    destinations_values = {d:sum(flow_values[i] for i in destinations_indices[d]) for d in destinations}

    # Construct new adjacency matrix
    new_adj_mat = []
    for i, row in enumerate(adj_mat):
        if i in destinations:
            new_adj_mat.append(row+[0, 1])
        else:
            new_adj_mat.append(row+[0, 0])
    new_adj_mat.append([1 if j in sources else 0 for j in range(len(adj_mat)+2)])
    new_adj_mat.append([0]*(len(adj_mat)+2))

    # Construct new transport times
    new_transport_times = []
    for i, row in enumerate(transport_times):
        if i in destinations:
            new_transport_times.append(row+[float("inf"), 1])
        else:
            new_transport_times.append(row+[float("inf"), float("inf")])
    new_transport_times.append([1 if j in sources else float("inf") for j in range(len(transport_times)+2)])
    new_transport_times.append([float("inf")]*(len(transport_times)+2))

    # Construct new aggregated multi flow with adding the super-source and super destination
    new_aggregated_flow = []
    for i, row in enumerate(aggregated_flow):
        if i in destinations:
            new_aggregated_flow.append(row+[0, destinations_values[i]])
        else:
            new_aggregated_flow.append(row+[0, 0])
    new_aggregated_flow.append([sources_values[j] if j in sources else 0 for j in range(len(aggregated_flow)+2)])
    new_aggregated_flow.append([0]*(len(aggregated_flow)+2))
    """new_multi_flow = [[] for _ in range(len(multi_flow))]
    for id_pair in range(len(multi_flow)):
        for i, row in enumerate(multi_flow[id_pair]):
            if i in destinations:
                new_multi_flow[id_pair].append(row+[0, destinations_values[i]])
            else:
                new_multi_flow[id_pair].append(row+[0, 0])
        new_multi_flow[id_pair].append([sources_values[j] if j in sources else 0 for j in range(len(multi_flow[id_pair])+2)])
        new_multi_flow[id_pair].append([0]*(len(multi_flow[id_pair])+2))"""

    return new_adj_mat, new_transport_times, new_aggregated_flow


def read_graph_instance(dir_name_graph_instance, id_instance, instance_type = "simulated"):
    if instance_type == "simulated":
        file_path_graph = dir_name_graph_instance+"instance_"+str(id_instance)+".npy"
        graph_instance = read_data(file_path_graph, read_all = True)
        arcs, nodes = graph_instance["arcs"], graph_instance["nodes"]
        adj_mat = graph_instance["adj_mat"]
        transport_times = graph_instance["transport_times"]
        pairs_org = graph_instance["pairs"]

    elif instance_type == "real":
        file_path = dir_name_graph_instance+"mat_cap_time_pair.npy"
        graph_instance = np.load(file_path, allow_pickle = True).flat[0]
        arcs, nodes = None, None
        adj_mat = graph_instance["mat"]
        transport_times = graph_instance["time"]

        row_conditions = adj_mat[0] == [0]*len(adj_mat)
        column_condition = [adj_mat[i][0] for i in range(len(adj_mat))]
        if row_conditions and column_condition:
            adj_mat = [row[1:] for row in adj_mat[1:]]
        
        row_conditions = transport_times[0] == [0]*len(transport_times)
        column_condition = [transport_times[i][0] for i in range(len(transport_times))]
        if row_conditions and column_condition:
            transport_times = [row[1:] for row in transport_times[1:]]

        pairs_org = graph_instance["pair"]

    else:
        print(instance_type, " instance type is not recognised.")
        sys.exit()

    return arcs, nodes, adj_mat, transport_times, pairs_org


def read_multif_flow_instance (dir_name_graph_instance, dir_name_multi_flow_instance, id_instance, 
                               instance_type = "simulated",
                               trans_func = False):
    opt_returns = None
    # Read the graph
    arcs, nodes, adj_mat, transport_times, pairs_org = read_graph_instance(dir_name_graph_instance, id_instance,
                                                                           instance_type = instance_type)
    pairs_org = [(int(pair[0]), int(pair[1])) for pair in pairs_org]
    
    # Read the associated multiflow
    if instance_type == "simulated":
        #file_path_multi_flow = dir_name_multi_flow_instance+"/RF_instance_"+str(id_instance)+".npy"
        file_path_multi_flow = os.path.join(dir_name_multi_flow_instance, "multi_flow_instance_"+str(id_instance)+".npy")

    elif instance_type == "real":
        file_path_multi_flow = os.path.join(dir_name_multi_flow_instance, "capacity"+str(id_instance)+".npy")

    else:
        print(instance_type, " instance type is not recognised.")
        sys.exit()
    
    data = np.load(file_path_multi_flow, allow_pickle = True).flat[0]
    all_multi_flow, flow_infos = data["matrice"], data["flow"]
    
    if "transition_functions" in data:
        opt_returns = {"transition_functions":data["transition_functions"]}
    if instance_type == "real":
        row_conditions = [multi_flow[0] == [0]*len(multi_flow) for multi_flow in all_multi_flow]
        column_conditions = [[multi_flow[i][0] for i in range(len(multi_flow))] == [0]*len(multi_flow) for multi_flow in all_multi_flow]

        if all(row_conditions) and all(column_conditions):
            all_multi_flow = [[row[1:] for row in multi_flow[1:]] for multi_flow in all_multi_flow]
        
        all_pairs = [(fl_inf[0], fl_inf[1]) for fl_inf in flow_infos]

        pts_interests = set([pair[0] for pair in all_pairs]+[pair[1] for pair in all_pairs])
        min_pts, max_pts = min(pts_interests), max(pts_interests)
        if max_pts >= len(all_multi_flow[0]):
            all_pairs = [(fl_inf[0] - 1, fl_inf[1] - 1) for fl_inf in flow_infos]

        if min_pts < 0:
            print("Problem in point of interests.")
            sys.exit()

        pts_interests = set([pair[0] for pair in pairs_org]+[pair[1] for pair in pairs_org])
        min_pts, max_pts = min(pts_interests), max(pts_interests)
        if max_pts >= len(all_multi_flow[0]):
            pairs_org = [(pair[0] - 1, pair[1] - 1) for pair in pairs_org] 
    
    else:
        all_pairs = [(fl_inf[0], fl_inf[1]) for fl_inf in flow_infos]
    all_flow_values = [fl_inf[2] for fl_inf in flow_infos]

    if all_pairs != pairs_org or len(all_multi_flow[0]) != len(adj_mat) or len(all_multi_flow[0][0]) != len(adj_mat) \
        or len(pairs_org) != len(all_flow_values):
        print(all_pairs == pairs_org)
        print("all pairs ")
        print(all_pairs)
        print("org pairs ")
        print(pairs_org)
        sys.exit()
        print(all_pairs[0], all_pairs[1])
        print(pairs_org[0], pairs_org[1])
        print("len vs len ", len(all_multi_flow[0]), len(adj_mat))
        print("Other legths", len(adj_mat), len(transport_times))
        print(" acces ", adj_mat[0])
        print("Pairs do not correspond or problem in length of 'multi_flow'.")
        sys.exit()
    
    flow_values, pairs, multi_flow = [], [], []
    for i in range(len(all_flow_values)):
        if all_flow_values[i] != 0:
            flow_values.append(all_flow_values[i])
            pairs.append(all_pairs[i])
            multi_flow.append(all_multi_flow[i])
    
    return adj_mat, transport_times, pairs, flow_values, multi_flow, arcs, nodes, opt_returns


def read_ntreat_instance (dir_name_graph_instance, 
                          dir_name_multi_flow_instance, 
                          id_instance = None, 
                          instance_type = "simulated",
                          add_supsource_supdestination = True,
                          trans_func = False):
    adj_mat, transport_times, pairs, flow_values, multi_flow, arcs, nodes, opt_returns = read_multif_flow_instance (dir_name_graph_instance,
                                                                                                                    dir_name_multi_flow_instance, 
                                                                                                                    id_instance, 
                                                                                                                    instance_type = instance_type,
                                                                                                                    trans_func = trans_func)

    # Process the aggregated flow
    aggregated_flow = [[sum(multi_flow[i][u][v] for i in range(len(multi_flow))) for v in range(len(multi_flow[0]))] for u in range(len(multi_flow[0]))]
    
    # Verif the adjacency matrix 'adj_mat'
    non_existing_arcs = []
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat)):
            if adj_mat[i][j] == 0 and aggregated_flow[i][j] > 0:
                non_existing_arcs.append((i, j))
    
    if len(non_existing_arcs) > 0:
        print("Flow exists in non existant arc.", sorted(non_existing_arcs))
        sys.exit()
    
    # Take only the arcs where there is flow, update 'adj_mat' and 'transport_times'
    adj_mat = [[adj_mat[i][j] if aggregated_flow[i][j] > 0 else 0 for j in range(len(adj_mat))] for i in range(len(adj_mat))]
    transport_times = [[transport_times[i][j] if aggregated_flow[i][j] > 0 else float("inf") for j in range(len(transport_times))]\
                                                                                                for i in range(len(transport_times))]
    
    # Verif flow val
    sources, destinations = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    if not (set(sources) & set(destinations)):
        sum_source_out_flow = 0
        for s in set(sources):
             #print("ss ", sum(aggregated_flow[s][v] for v in range(len(aggregated_flow))))
             #print("se ", sum(aggregated_flow[v][s] for v in range(len(aggregated_flow))))
             sum_source_out_flow += sum(aggregated_flow[s][v] for v in range(len(aggregated_flow))) - sum(aggregated_flow[v][s] for v in range(len(aggregated_flow)))
        sum_destination_in_flow = 0
        for t in set(destinations):
            sum_destination_in_flow += sum(aggregated_flow[v][t] for v in range(len(aggregated_flow))) - sum(aggregated_flow[t][v] for v in range(len(aggregated_flow)))
        sum_flow_val = sum(flow_values)
        if sum_flow_val != sum_source_out_flow or sum_source_out_flow != sum_destination_in_flow or sum_source_out_flow <= 0 or sum_destination_in_flow <= 0:
            print("Problem in flow values (source/destination) ", sum_source_out_flow, sum_destination_in_flow, sum_flow_val)
            sys.exit()

    # Preprocess instance for the desaggregation algorithm 
    # (by adding a dimension for the super source and another one for the super destination in the reelevent matrices)
    if add_supsource_supdestination:
        adj_mat, transport_times, aggregated_flow = process_instance (adj_mat = adj_mat, 
                                                                        transport_times = transport_times, 
                                                                        aggregated_flow = aggregated_flow,
                                                                        pairs = pairs,
                                                                        flow_values = flow_values)
    
    # Return the complete instance
    complete_instance = {}
    complete_instance["arcs"] = arcs
    complete_instance["nodes"] = nodes
    complete_instance["adj_mat"] = adj_mat
    complete_instance["transport_times"] = transport_times
    complete_instance["pairs"] = pairs
    complete_instance["flow_values"] = flow_values
    complete_instance["aggregated_flow"] = aggregated_flow
    complete_instance["multi_flow"] = multi_flow
    if opt_returns is not None and "transition_functions" in opt_returns:
        complete_instance["transition_functions"] = opt_returns["transition_functions"]
    return complete_instance