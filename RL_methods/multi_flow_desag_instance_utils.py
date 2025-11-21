import sys
from copy import deepcopy
from utils.pre_process_utils import read_ntreat_instance


##############################################  Functions  ############################################## 
def init_random_transition_matrices(mfd_instance):
    mfd_instance.original_transition_function = {(u, v):{(v,w):mfd_instance.adj_mat[v][w] 
                                                                    for w in range(len(mfd_instance.adj_mat))}
                                                                        for u in range(len(mfd_instance.adj_mat)) 
                                                                            for v in range(len(mfd_instance.adj_mat)) 
                                                                                if mfd_instance.adj_mat[u][v] == 1}
    mfd_instance.original_transition_from_sources = {s:{(s,w):mfd_instance.adj_mat[s][w] 
                                                                            for w in range(len(mfd_instance.adj_mat))}
                                                                                for s in mfd_instance.original_transition_from_sources}
    mfd_instance.original_transition_to_destinations = {d:{(v,d):mfd_instance.adj_mat[v][d] 
                                                                            for v in range(len(mfd_instance.adj_mat))}
                                                                                for d in mfd_instance.original_transition_to_destinations}
    mfd_instance.transition_function = deepcopy(mfd_instance.original_transition_function)
    mfd_instance.transition_from_sources = deepcopy(mfd_instance.transition_from_sources)
    mfd_instance.transition_to_destinations = deepcopy(mfd_instance.transition_to_destinations)



def init_support_transition_matrices(mfd_instance):
    mfd_instance.original_transition_function = {key:{key2:int(bool(mfd_instance.original_transition_function[key][key2] > 0))
                                                for key2 in mfd_instance.original_transition_function[key]}
                                                    for key in mfd_instance.original_transition_function}
    mfd_instance.original_transition_from_sources = {key:{key2:int(bool(mfd_instance.original_transition_from_sources[key][key2] > 0))
                                                    for key2 in mfd_instance.original_transition_from_sources[key]}
                                                        for key in mfd_instance.original_transition_from_sources}
    mfd_instance.original_transition_to_destinations = {key:{key2:int(bool(mfd_instance.original_transition_to_destinations[key][key2] > 0))
                                                        for key2 in mfd_instance.original_transition_to_destinations[key]} 
                                                            for key in mfd_instance.original_transition_to_destinations}
    mfd_instance.transition_function = deepcopy(mfd_instance.original_transition_function)
    mfd_instance.transition_from_sources = deepcopy(mfd_instance.transition_from_sources)
    mfd_instance.transition_to_destinations = deepcopy(mfd_instance.transition_to_destinations)



def construct_instances (dir_name_graph_instance, 
                        dir_name_multi_flow_instance,
                        nb_instances,
                        ls_update_transport_time,
                        ls_update_transition_functions):
    dict_instances = {}
    for i in range(nb_instances):
        print("Construct instance ", i)
        # Reading instance
        complete_instance = read_ntreat_instance (dir_name_graph_instance = dir_name_graph_instance, 
                                                  dir_name_multi_flow_instance = dir_name_multi_flow_instance,
                                                  id_instance = i,
                                                  add_supsource_supdestination = False,
                                                  trans_func = True)
        # Unpacking of instances
        adj_mat = complete_instance["adj_mat"]
        aggregated_flow = complete_instance["aggregated_flow"]
        transport_times = complete_instance["transport_times"]
        pairs = complete_instance["pairs"]
        flow_values = complete_instance["flow_values"]
        aggregated_flow = complete_instance["aggregated_flow"]
        multi_flow = complete_instance["multi_flow"]
        ls_transition_function = complete_instance["transition_functions"]
        # Create mfd instance for each value in 'ls_update_transport_time' and 'ls_update_transition_functions' 
        for update_transport_time in ls_update_transport_time:
            for update_transition_function in ls_update_transition_functions:
                mfd_instance = MultiFlowDesagInstance(adj_mat, 
                                                     aggregated_flow,
                                                     transport_times, 
                                                     pairs, 
                                                     flow_values, 
                                                     ls_transition_function,
                                                     update_transport_time = update_transport_time,
                                                     update_transition_functions = update_transition_function)
                dict_instances[(i, update_transport_time, update_transition_function)] = (mfd_instance, multi_flow)
    return dict_instances



##############################################  Classes   ##############################################
class MultiFlowDesagInstance:

    def __init__(self, 
                adj_mat, 
                aggregated_flow,
                transport_times, 
                pairs, 
                flow_values, 
                ls_transition_function,
                update_transport_time = False,
                update_transition_functions = False):
        
        # Check for zero flow values
        for f_val in flow_values:
            if f_val == 0:
                print("There is a zero flow value.")
                print(flow_values)
                sys.exit()
         # Cash the original adjacency matrix
        self.original_adj_mat = adj_mat
        self.adj_mat = deepcopy(adj_mat)

        # The aggregated flow and unattributed flow
        self.original_aggregated_flow = aggregated_flow
        self.aggregated_flow = deepcopy(aggregated_flow)

        # Create time related informations
        self.ideal_transport_times = transport_times
        self.transport_times = deepcopy(transport_times)
        self.original_update_transport_time = update_transport_time
        self.update_transport_time = update_transport_time

        # The source-destination pairs along with the number of excluded pairs
        self.pairs = deepcopy(pairs)

        # The flow values associated to the pairs
        self.original_flow_values = flow_values

        # Set the transition function
        self.original_transition_function = ls_transition_function[0]
        self.original_transition_from_sources = ls_transition_function[1]
        self.original_transition_to_destinations = ls_transition_function[2]
        self.transition_function = deepcopy(ls_transition_function[0])
        self.transition_from_sources = deepcopy(ls_transition_function[1])
        self.transition_to_destinations = deepcopy(ls_transition_function[2])

        # True iif we want to update the transition functions
        self.original_update_transition_functions = update_transition_functions
        self.update_transition_functions = update_transition_functions

    
    def reset_instance(self):
        # Adjacency matrix reset
        self.adj_mat = deepcopy(self.original_adj_mat)

        # The aggregated flow and unattributed flow reset
        self.aggregated_flow = deepcopy(self.original_aggregated_flow)

        # Transport times reset
        self.transport_times = deepcopy(self.ideal_transport_times)
        self.update_transport_time = self.original_update_transport_time
        
        # reset the transition function
        self.transition_function = deepcopy(self.original_transition_function)
        self.transition_from_sources = deepcopy(self.original_transition_from_sources)
        self.transition_to_destinations = deepcopy(self.original_transition_to_destinations)
        self.update_transition_functions = self.original_update_transition_functions


    def return_update_time_target (self, node1, node2): 
        return self.ideal_transport_times[node1][node2] * (self.original_aggregated_flow[node1][node2] / self.aggregated_flow[node1][node2])
    

    def update_flow_infos(self, info_pair, multi_flow, generated_flow_values):
        # (indice of the pair in 'self.pairs', the chosen path for the pair, capacity of the path, remaining_flow_value of the pair)
        path, flow_amount, ind_pair = info_pair[1], info_pair[2], info_pair[0] 
        # Update the flow value associated to the chosen pair
        info_pair[3] -= flow_amount
        generated_flow_values[ind_pair] += flow_amount

        #  Update the flow/adjacency matrix at the rest of the path
        for i in range(0, len(path)-1):
            # Decrease the flow on the current arc of the aggregated flow 
            self.aggregated_flow[path[i]][path[i+1]] -= flow_amount
            # Augment the flow on the current arc of of the desagregated constructed flow (if the associated pair is known)
            multi_flow[ind_pair][path[i]][path[i+1]] += flow_amount
            # Set the current arc to 0 if the aggregated flow on this arc is 0 
            self.adj_mat[path[i]][path[i+1]] = int(bool(self.aggregated_flow[path[i]][path[i+1]]) != 0)
            # Update the time on the current arc if it is still there
            if self.update_transport_time:
                self.transport_times[path[i]][path[i+1]] = self.return_update_time_target(path[i], path[i+1]) if self.adj_mat[path[i]][path[i+1]] == 1 else float("inf")
            # Update transition function if needed
            if self.update_transition_functions:
                if i == 0:
                    self.transition_from_sources[path[0]][(path[0], path[1])] -= flow_amount
                else:
                    self.transition_function[(path[i-1], path[i])][(path[i], path[i+1])] -= flow_amount
            
            # Sanity check
            if self.aggregated_flow[path[i]][path[i+1]] < 0:
                print("Curent arc ", path[i], path[i+1], self.aggregated_flow[path[i]][path[i+1]])
                print("Capacity is negative.", type(self).__name__)
                sys.exit() 

        # Update the flow on the transition list associated to the destination if 'update_transition_functions' is enabled
        if self.update_transition_functions: self.transition_to_destinations[path[-1]][(path[-2], path[-1])] -= flow_amount


    def update_remaining_pairs(self, dict_infos_rem_pairs_paths, dict_rem_ind_pairs):
        for ind_pair in list(dict_rem_ind_pairs.keys()):
            pair = dict_rem_ind_pairs[ind_pair]
            if dict_infos_rem_pairs_paths[pair][3] == 0:
                del dict_infos_rem_pairs_paths[pair]
                del dict_rem_ind_pairs[ind_pair]