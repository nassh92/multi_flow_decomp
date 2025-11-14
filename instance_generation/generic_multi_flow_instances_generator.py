import numpy as np

import pprint

import os

import sys

import random

import itertools

from copy import deepcopy

# 'C:\\Users\\HADDAM\\Documents\\Python Scripts\\multi_flow_decomp\\'
sys.path.append(os.getcwd())
from utils.shortest_path_solvers import DijkstraShortestPathsSolver

from instance_generation.pairs_utils import process_weight_pairs

from instance_generation.random_instance_generator import read_data, generate_random_instance

from instance_generation.transition_function_utils import construct_transition_functions, update_transition_functions

from utils.graph_utils import has_arc, delete_arc, init_graph_arc_attribute_vals, is_adjacency_matrix



def update_time (transport_times, 
                 og_transport_times, 
                 capacities, 
                 og_capacities, arc):
    # Update the traversal time of 'arc' depending on its remaning capacity and its original traversal time 
    transport_times[arc[0]][arc[1]] = og_transport_times[arc[0]][arc[1]] * (og_capacities[arc[0]][arc[1]] / capacities[arc[0]][arc[1]])



def generate_multi_flow_instance(graph, 
                                 capacities, 
                                 transport_times, 
                                 pairs, 
                                 desired_flow_values, 
                                 min_fl = 1, 
                                 nb_it_print = None,
                                 pairs_generation = "capacity",
                                 weight_pairs = None,
                                 predecessors_list = None,
                                 return_transition_function = False):
    """
        Generate a multiflow from a given nework represented by its adjacency matrix, its capacities matrix and its traversal times matrix 
    """
    # Store the original graph matrices (the capacity matrix and the traversal times matrix)
    og_capacities = deepcopy(capacities)
    og_transport_times = deepcopy(transport_times)

    # Process the weights associated to each pair
    if weight_pairs is None: 
        weight_pairs = process_weight_pairs(pairs, 
                                            graph, 
                                            arc_attribute_vals = capacities,
                                            predecessors_list = predecessors_list, 
                                            pairs_generation = pairs_generation)

    # Initializations
    cpt_saturated, nb_it = 0, 0
    generated_flow_values = [0 for _ in range(len(pairs))]
    multi_flow = [init_graph_arc_attribute_vals(graph, init_val = 0) for _ in range(len(pairs))]

    # If 'return_transition_function' is True, construct the transition functions
    if return_transition_function:
        trans_func, trans_from_sources, trans_to_destinations = construct_transition_functions(graph, 
                                                                                               pairs,
                                                                                               predecessors_list = predecessors_list)

    # Loop until all paris are saturated
    while cpt_saturated < len(pairs):
        # Choose a pair stochastically depending of the sum of capacity of the arcs incident on its endpoints
        id_pair = random.choices(population = list(range(len(pairs))), 
                                 weights = weight_pairs,
                                 k = 1)[0]
        source, destination = pairs[id_pair]

        # Create a DijkstraShortestPathsSolver instance and solve it (by running the dijkstra algorithm) 
        dijkstra_solver = DijkstraShortestPathsSolver(source = source,
                                                      graph = graph,
                                                      weights = transport_times, 
                                                      mode = "min_distance",
                                                      optional_infos = {"predecessors_list":predecessors_list})
        dijkstra_solver.run_dijkstra()
        
        # For the selected pair, augment the flow along a path from the source of the pair to its destination if there is a path
        if dijkstra_solver.path_estimates[destination] != float("inf"):
            # Return a random path from the source to the destination
            path = dijkstra_solver.return_path(destination, 
                                               path_type = "random")
            
            # Compute the flow amount as the minimum the capacity of the selected path, the remaining flow value of the pair and 'min_fl'
            path_capacity = min(capacities[path[i]][path[i+1]] for i in range(len(path)-1))
            fl_amount = min(path_capacity, desired_flow_values[id_pair], min_fl)

            # Increase the flow along the arcs of the selected path by 'fl_amount' units for the corresponding pair and 
            # decrease the capacities of those arcs
            if return_transition_function: # initialization fo the case where 'return_transition_function' is enabled 
                arc1 = None
            for i in range(len(path)-1):
                # Unpack the arc
                u, v = path[i], path[i+1]
                # Update the transition function if 'return_transition_function' is enabled
                if return_transition_function:
                    arc2 = (u, v) 
                    update_transition_functions((trans_func, trans_from_sources, trans_to_destinations),
                                                arc1, arc2, fl_amount, pairs)
                    arc1 = (arc2[0], arc2[1])
                # Augment the flow on the generated multiflow and reduce the capacities by the same amount
                multi_flow[id_pair][u][v] += fl_amount
                capacities[u][v] -= fl_amount
                
                # If the capacity of an arc of the path reaches 0, delete it from the graph
                if capacities[u][v] == 0:  
                    delete_arc(graph, u, v)
                else: # else update the transport time of this arc
                    update_time (transport_times, og_transport_times, capacities, og_capacities, (u, v))
            # Update the transition function if 'return_transition_function' is enabled
            if return_transition_function is not None:
                update_transition_functions((trans_func, 
                                             trans_from_sources, 
                                             trans_to_destinations),
                                            arc2, None, fl_amount, pairs)
            
            # Decrease the remaning flow value of the selected pair, and increase the generated flow value associated to the same pair
            desired_flow_values[id_pair] -= fl_amount
            generated_flow_values[id_pair] += fl_amount

            # If the remaning flow value of the selected pair reaches zero, set its weight (and thus probability) to zero and
            # increment the number of saturated pairs 
            if desired_flow_values[id_pair] == 0:
                weight_pairs[id_pair] = 0
                cpt_saturated += 1
            
        else: # else increment the number of saturated pairs and set the weight (probability) of the corresponding pair to zero
            weight_pairs[id_pair] = 0
            cpt_saturated += 1
        
        # Print
        if nb_it_print is not None and nb_it % nb_it_print == 0:
            print(nb_it, cpt_saturated, sum(generated_flow_values))

        # Increment the number of iterations
        nb_it += 1

    # Construct the return dict and return it
    return_dict = {}
    return_dict["multi_flow"] = multi_flow
    return_dict["flow_values"] = generated_flow_values
    if return_transition_function:
        return_dict["transition_functions"] = (trans_func, trans_from_sources, trans_to_destinations)
    return return_dict



def construct_multi_flow_instances_from_saved_graph_instances(dir_data_name,
                                                              dir_save_name, 
                                                              desired_flow_values = None,
                                                              min_fl = 1, 
                                                              nb_it_print = None,
                                                              pairs_generation = "capacity",
                                                              return_transition_function = False):
    """
        Generate a number of multiflow instances from network instances files contained in directory 'dir_name'
        Note : The 'weight_pairs' returned by the call 'read_data' is ignored,
        instead the weight_pairs used is the one produced during the multiflow generation 
        by the call of the function 'generate_multi_flow_instance'.
    """
    file_paths = [os.path.join(dir_data_name, f) for f in os.listdir(dir_data_name) 
                                                    if os.path.isfile(os.path.join(dir_data_name, f)) and\
                                                        f[-4:] != ".txt"]
    
    for f_path in file_paths:
        num_instance = f_path.split("_")[-1][:-4]
        print("Treating instance number ", num_instance)
        adj_mat, capacities, transport_times, pairs, weight_pairs = read_data(f_path, 
                                                                              read_all = False)

        if desired_flow_values is None: # If the desired flow values is not None we take the default value of the 'desired_flow_values' 
            desired_flow_values = [float("inf") for _ in range(len(pairs))]

        return_dict = generate_multi_flow_instance(adj_mat, capacities, 
                                                   transport_times, 
                                                   pairs, desired_flow_values, 
                                                   min_fl = min_fl, 
                                                   nb_it_print = nb_it_print,
                                                   pairs_generation = pairs_generation,
                                                   return_transition_function = return_transition_function)
        saved_dict = {"matrice":return_dict["multi_flow"],
                      "flow":[(pairs[i][0], pairs[i][1], return_dict["flow_values"][i]) for i in range(len(pairs))],
                      "transition_functions":None if "transition_functions" not in return_dict else return_dict["transition_functions"]}
        np.save(os.path.join(dir_save_name, "multi_flow_instance_"+num_instance), saved_dict)



def construct_complete_multi_flow_instances(dir_save_name, 
                                            dict_graph_params,
                                            desired_flow_values = None,
                                            min_fl = 1, 
                                            nb_it_print = None,
                                            pairs_generation = "capacity",
                                            return_transition_function = False):
    """
        Generate a number of multiflow instances by first constructing a network instances and then the multiflow instances 
    """
    # Generate the graph instances
    ls_graph_instances = []
    for num_instance in range(dict_graph_params["nb_instances"]):
        print("Generate graph instance ", num_instance)
        # Return infos of the generated intance
        infos_instance = generate_random_instance(nb_nodes = dict_graph_params["nb_nodes"], 
                                                  grid_size = dict_graph_params["grid_size"], 
                                                  r = dict_graph_params["r"], 
                                                  nb_arcs = dict_graph_params["nb_arcs"], 
                                                  max_nb_draws_gen_graph = dict_graph_params["max_nb_draws_gen_graph"], 
                                                  max_nb_tries_gen_graph = dict_graph_params["max_nb_tries_gen_graph"],
                                                  base_capacity = dict_graph_params["base_capacity"], 
                                                  capacity_factor = dict_graph_params["capacity_factor"], 
                                                  transport_time_fraction = dict_graph_params["transport_time_fraction"], 
                                                  nb_pairs = dict_graph_params["nb_pairs"], 
                                                  nb_max_draws_pairs = dict_graph_params["nb_max_draws_pairs"], 
                                                  distance_type = dict_graph_params["distance_type"], 
                                                  pairs_generation = dict_graph_params["pairs_generation"], 
                                                  pairs_criteria = dict_graph_params["pairs_criteria"], 
                                                  print_ = dict_graph_params["print_"])
        ls_graph_instances.append(infos_instance)    
    print("Flow generation.")
    # Generate the multiflow instance for each graph instance and save it 
    for id_graph_instance in range(len(ls_graph_instances)):
        print("Construct multi flow for instance ", id_graph_instance)
        # Unpack the instance
        graph_instance = ls_graph_instances[id_graph_instance]
        adj_mat = graph_instance["adj_mat"]
        capacities =  graph_instance["capacities"]
        transport_times = graph_instance["transport_times"]
        pairs = graph_instance["pairs"]

        # If the desired flow values is not None we take the default value of the 'desired_flow_values'
        if desired_flow_values is None:  
            desired_flow_values = [float("inf") for _ in range(len(pairs))]

        # Generate the multiflow instance
        return_dict = generate_multi_flow_instance(adj_mat, capacities,
                                                   transport_times, 
                                                   pairs, desired_flow_values, 
                                                   min_fl = min_fl, 
                                                   nb_it_print = nb_it_print,
                                                   pairs_generation = pairs_generation,
                                                   return_transition_function = return_transition_function)
        
        np.save(os.path.join(dir_save_name, "multi_flow_instance_"+str(id_graph_instance)), return_dict)



###########################################  Just look up functions  ###########################################
def read_transition_functions(dir_data_name, num_instance):
    data = np.load(os.path.join(dir_data_name, "multi_flow_instance_"+str(num_instance)+".npy"), allow_pickle = True).flat[0]
    transition_functions = data["transition_functions"]
    transition_func, transition_from_sources, transition_to_destinations = transition_functions[0], transition_functions[1], transition_functions[2]
    
    print("Transition from sources ")
    print(transition_from_sources)

    print("Transition function ")
    print(transition_func)



def return_statistics(dir_data_name):
    statistics_transition_functions = {}
    file_paths = [os.path.join(dir_data_name, f) for f in os.listdir(dir_data_name) if os.path.isfile(os.path.join(dir_data_name, f))]
    for f_path in file_paths:
        num_instance = f_path.split("_")[-1][:-4]
        data = np.load(f_path, allow_pickle = True).flat[0]
        multi_flow = data["matrice"]
        transition_func, transition_from_sources, transition_to_destinations = data["transition_functions"][0], data["transition_functions"][1], data["transition_functions"][2]
        elem = []

        trans_func_stats, trans_func_stats2, trans_func_stats3 = [], [], []
        for arc in transition_func:
            sum_weights = sum(transition_func[arc].values())
            if sum_weights != 0:
                probabilities = [weight/sum_weights for weight in transition_func[arc].values()]
                mean_ = np.mean(probabilities)
                std_ = sum(abs(proba - mean_) for proba in probabilities)/len(probabilities)
                trans_func_stats3.append(max(probabilities) - min(probabilities))
                trans_func_stats.append(std_)
                trans_func_stats2.append(std_/mean_)

        elem.append(np.mean(trans_func_stats3))    
        elem.append(np.mean(trans_func_stats))
        elem.append(np.mean(trans_func_stats2))

        trans_from_sources_stats = []
        for source in transition_from_sources:
            sum_weights = sum(transition_from_sources[source].values())
            if sum_weights != 0:
                probabilities = [weight/sum_weights for weight in transition_from_sources[source].values()]
                mean_ = np.mean(probabilities)
                trans_from_sources_stats.append(sum(abs(proba - mean_) for proba in probabilities)/len(probabilities))
            
        elem.append(np.mean(trans_from_sources_stats))

        trans_to_destinations_stats = []
        for destination in transition_to_destinations:
            sum_weights = sum(transition_to_destinations[destination].values())
            if sum_weights != 0:
                probabilities = [weight/sum_weights for weight in transition_to_destinations[destination].values()]
                mean_ = np.mean(probabilities)
                trans_to_destinations_stats.append(sum(abs(proba - mean_) for proba in probabilities)/len(probabilities))
            
        elem.append(np.mean(trans_to_destinations_stats))
        
        aggregated_flow = [[sum(multi_flow[i][u][v] for i in range(len(multi_flow))) for v in range(len(multi_flow[0]))] for u in range(len(multi_flow[0]))]
        mean_out_deg = sum(sum(e > 0 for e in row) for row in aggregated_flow)/len(aggregated_flow)
        mean_in_deg = sum(sum(aggregated_flow[i][j]>0 for i in range(len(aggregated_flow))) for j in range(len(aggregated_flow)))/len(aggregated_flow)

        elem.append(mean_out_deg)
        elem.append(mean_in_deg)

        statistics_transition_functions[num_instance] = elem
    
    return statistics_transition_functions

################################################################################################################################



def main():
    test_names = {"generate_multi_flow_instances_from_saved_graph_instances",
                  "multi_generate_multi_flow_instances_from_saved_graph_instances", 
                  "generate_complete_multi_flow_instances",
                  "read_transition_func",
                  "process_statistics"}

    test_name = "multi_generate_multi_flow_instances_from_saved_graph_instances"

    if test_name not in test_names:
        print("Test name unrecognized.")
        sys.exit()

    if test_name == "generate_multi_flow_instances_from_saved_graph_instances":
        construct_multi_flow_instances_from_saved_graph_instances(dir_data_name = "instance_generation/instances/capacity/",
                                                                  dir_save_name = "multi_flow_generation/transition_function_instances/", 
                                                                  return_transition_function = True)
     
    elif test_name == "multi_generate_multi_flow_instances_from_saved_graph_instances":
        nb_nodes_ls = [55, 75, 95]
        nb_pairs_ls = [10, 15, 20]
        params_names = list(itertools.product(nb_nodes_ls, nb_pairs_ls))
        dir_save_names = ["nb_nodes="+str(params_names[i][0])+"_pairs="+str(params_names[i][1])
                                    for i in range(len(params_names))]
        for i in range(len(params_names)):
            if dir_save_names[i] != "nb_nodes=75_pairs=15":
                nb_nodes, nb_pairs = params_names[i]
                print("--------------------------------------------------------------------------")
                print(nb_nodes, nb_pairs)
                print("--------------------------------------------------------------------------")
                dir_data_name = "instance_generation/instances/capacity/"+"instances_nbnodes="+str(nb_nodes)+"_pairs="+str(nb_pairs)+"/"
                dir_save_name = "multi_flow_generation/transition_function_instances/"+dir_save_names[i]+"/"
                construct_multi_flow_instances_from_saved_graph_instances(
                                                                dir_data_name = dir_data_name,
                                                                dir_save_name = dir_save_name, 
                                                                return_transition_function = True)

    elif test_name == "generate_complete_multi_flow_instances":
        # Create dictionary of parameters
        dict_graph_params = {}
        dict_graph_params["nb_instances"] = 100
        dict_graph_params["nb_nodes"] = 50
        dict_graph_params["grid_size"] = 100
        dict_graph_params["r"] = 25
        dict_graph_params["nb_arcs"] = 200
        dict_graph_params["max_nb_draws_gen_graph"] = 2000
        dict_graph_params["max_nb_tries_gen_graph"] = 3
        dict_graph_params["base_capacity"] = 10
        dict_graph_params["capacity_factor"] = 4
        dict_graph_params["transport_time_fraction"] = 0.25
        dict_graph_params["nb_pairs"] = 10
        dict_graph_params["nb_max_draws_pairs"] = 100
        dict_graph_params["distance_type"] = "euclidean"
        dict_graph_params["pairs_generation"] = "capacity"
        dict_graph_params["pairs_criteria"] = 0
        dict_graph_params["print_"] = False

        # Construct the mulfi flow instances
        construct_complete_multi_flow_instances(dir_save_name = "multi_flow_generation/temp/",
                                                dict_graph_params = dict_graph_params,
                                                return_transition_function = True)
        
    elif test_name == "read_transition_func":
        read_transition_functions(dir_data_name = "multi_flow_generation/transition_function_instances/", 
                                  num_instance = 1)
        
    elif test_name == "process_statistics":
        stats_instances = return_statistics("multi_flow_generation/transition_function_instances/")
        print("------------------------------- Degree - min, max -------------------------------")
        print("Min out degree ", min(perf[-2] for perf in stats_instances.values()))
        print("mean out degree ", np.mean([perf[-2] for perf in stats_instances.values()]))
        print("Max out degree ", max(perf[-2] for perf in stats_instances.values()))
        print("Min in degree ", min(perf[-1] for perf in stats_instances.values()))
        print("Max in degree ", max(perf[-1] for perf in stats_instances.values()))
        print("------------------------------- By range - min, max -------------------------------")
        print("Minimal range ", min(perf[0] for perf in stats_instances.values()))
        print("Maximal range ", max(perf[0] for perf in stats_instances.values()))
        print("------------------------------- By mean absolute error -------------------------------")
        print("Minimal MAE ", min(perf[1] for perf in stats_instances.values()))
        print("Maximal MAE ", max(perf[1] for perf in stats_instances.values()))
        """
        pprint.pprint(sorted([(id_instance, perf[0], perf[1], perf[2]) for id_instance, perf in stats_instances.items()],
                             key = lambda x : (x[1], x[2], x[3])))
        pprint.pprint(sorted([(id_instance, perf[0], perf[1], perf[2]) for id_instance, perf in stats_instances.items()],
                             key = lambda x : (x[2], x[1], x[3])))
        """

if __name__ == "__main__":
    main()