import sys
import random
from copy import deepcopy
from abc import ABC, abstractmethod
import os
sys.path.append(os.getcwd())
from msmd.path_selectors import PATH_SELECTOR_TYPES, PathSelectorTransFunc
from msmd.multi_flow_desag_solver_utils import (PAIRS_CRITERIAS,
                                 PATH_CARDINALITY_CRITERIAS,
                                 STOCHASTIC_PAIRS_POLICIES_NAMES,
                                 SEMI_STOCHASTIC_PAIRS_POLICIES_NAMES,
                                 WeightPairsPolicy, 
                                 SemiStochasticReoderer, 
                                 PathFilterer)
from msmd.path_selectors import RandomPathSelector
from msmd.subgraph_constructors import SubGraphBestPathsConstructor
from utils.graph_utils import successors, get_arcs, create_isolated_nodes_graph, init_graph_arc_attribute_vals



###########################################################################################################################################################  
### A Multi Flow Desaggregation Class based which uses a transition function to selects paths
###########################################################################################################################################################

class MultiFlowDesagSolver():

    def __init__(self, 
                 mfd_instance,
                 max_path_length = 1000, 
                 total_nb_iterations = 1000,
                 max_nb_tries_find_path = 5,
                 maximal_flow_amount = float("inf"),
                 reodering_pairs_policy_name = None,
                 path_selector_type = "min_time_based",
                 construct_trans_function = False,
                 exclude_chosen_nodes = False,
                 ignore_conflicts = False,
                 graph_representation = "adjacency_matrix"):
        # Set the multi flow desaggregation instance
        self.mfd_instance = mfd_instance
        self.generated_flow_values = [0]*len(mfd_instance.original_flow_values)
        # Maximal flow amount to be subtracted from the graph on each iteration
        self.maximal_flow_amount = maximal_flow_amount
        # Total number of tries the algorithm tries to chose a path
        self.total_nb_iterations = total_nb_iterations
        # Number of consecutive tries for choosing a path
        self.max_nb_tries_find_path = max_nb_tries_find_path
        self.nb_tries_find_path = 0
        
        # The pair reoderer used to reorder the pairs in case we select a path for each of the pairs
        self.reodering_pairs_policy_name = reodering_pairs_policy_name
        if reodering_pairs_policy_name is None:
            self.pair_reorderer = None
        
        elif reodering_pairs_policy_name in STOCHASTIC_PAIRS_POLICIES_NAMES:
            self.pair_reorderer = WeightPairsPolicy(reodering_pairs_policy_name, self.mfd_instance.original_flow_values)
        
        elif reodering_pairs_policy_name in SEMI_STOCHASTIC_PAIRS_POLICIES_NAMES:
            self.pair_reorderer = SemiStochasticReoderer(reodering_pairs_policy_name)

        else:
            print("Reorder pair name is unrecognized.")
            sys.exit()

        # Creation of the path filterer
        self.path_filterer = PathFilterer(ignore_conflicts)

        # Set the newly formed transition function
        if construct_trans_function:
            self.constructed_transition_function = {(u, v):{(v, w):0 for w in successors(self.mfd_instance.adj_mat, v)}
                                                                        for u, v in get_arcs(self.mfd_instance.adj_mat)}
        
        else:
            self.constructed_transition_function = None

        # True iff matrix representation are used for the data
        self.graph_representation = graph_representation

        # Create remainig attribute useful for path selection
        self.create_desaggregator_paths_attributes(path_selector_type, max_path_length, exclude_chosen_nodes)


    def create_desaggregator_paths_attributes(self, path_selector_type, max_path_length, exclude_chosen_nodes):
        # Set 'update_transition_functions' to False
        self.mfd_instance.update_transition_functions = False

        # Creation of a path selector
        if path_selector_type is None:
            print("None value is unsupported for path selector.")
            sys.exit()
        
        elif path_selector_type == "min_time_based" or path_selector_type == "max_capacity_based":
            self.path_selector = RandomPathSelector(path_selector_type, max_path_length, exclude_chosen_nodes)
        
        elif path_selector_type == "trans_func_based":
            print("Trans func path selectors are not supported by class MultiFLowDesagSolver.")
            sys.exit()

        elif path_selector_type[:2] == "rl":
            print("RL based path selectors are not supported by class MultiFLowDesagSolver.")
            sys.exit()

        elif path_selector_type not in PATH_SELECTOR_TYPES:
            print("Path selector type is unrecognized : ", path_selector_type)
            sys.exit()

        else:
            print("This path selector is not yet supported by MultiFLowDesagSolver.")
            sys.exit()

        # Set a subgraph filterer
        self.subg_constructor = SubGraphBestPathsConstructor(self.path_selector.path_selector_type,
                                                             self.mfd_instance,
                                                             graph_representation = self.graph_representation)
    

    ########################################################   Iteration related func   ##################################################
    def _init_iteration_num(self):
        self.num_it = 0
        self.nb_tries_find_path = 0

    
    def _update_iteration_num (self, increment_num_it = True, increment_cons_tries = True, reset_cons_tries = False):
        if increment_num_it: self.num_it += 1
        if increment_cons_tries: self.nb_tries_find_path += 1
        if reset_cons_tries: self.nb_tries_find_path = 0


    def _has_iterated_too_much (self):
        return self.num_it >= self.total_nb_iterations or self.nb_tries_find_path >= self.max_nb_tries_find_path


    ########################################################   Update Constructed trans func   ##################################################
    def _update_path_constructed_transition_function(self, info_pair):
        # (indice of the pair in 'self.pairs', the chosen path for the pair, capacity of the path, remaining_flow_value of the pair)
        path, flow_amount = info_pair[1], info_pair[2]
        #  Update the flow/adjacency matrix at the rest of the path
        for i in range(1, len(path)-1):
            # Update the constructed transition function
            self.constructed_transition_function[(path[i-1], path[i])][(path[i], path[i+1])] += flow_amount

    
    def update_constructed_transition_function(self, selected_pairs, dict_infos_rem_pairs_paths):
        for pair in selected_pairs:
            self._update_path_constructed_transition_function(info_pair = dict_infos_rem_pairs_paths[pair])
    

    def update_multi_flow_network(self, multi_flow, selected_pairs, dict_infos_rem_pairs_paths):
        for pair in selected_pairs:
            self.mfd_instance.update_flow_infos(info_pair = dict_infos_rem_pairs_paths[pair], 
                                                multi_flow = multi_flow,
                                                generated_flow_values = self.generated_flow_values)
    
    
    ########################################################   Paths methods   ########################################################
    def select_path_pair_max_rem_flow (self, dict_infos_rem_pairs, dict_rem_ind_pairs):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair, 
        # capacity of the path, remaining_flow_value of the pair), ...}  
        # A 'dict' which has as key the index the pairs in 'self.pairs' and as value the corrsponding pair.
        # {indice of the pair in 'self.pairs' : the associated pair , ...}
        while len(dict_infos_rem_pairs) > 0 and not self._has_iterated_too_much ():
            max_remaining_flow_val = max(info_pair[3] for info_pair in dict_infos_rem_pairs.values())
            pairs_max_remaining_flow_val = [pair for pair in dict_infos_rem_pairs if dict_infos_rem_pairs[pair][3] == max_remaining_flow_val]
            source, destination = pairs_max_remaining_flow_val[random.randint(0, len(pairs_max_remaining_flow_val)-1)]
            
            # Construct a 'DijkstraShortestPathsSolver' instance for the source in the selected pair
            subg_s_d = self.subg_constructor.subgraph_source_dest(source, 
                                                                destination)
            # Return the candidate pair if there is a path linking the source to the destination and remove the pair otherwise 
            if not self.subg_constructor._exist_path ():
                ind_pair = dict_infos_rem_pairs[(source, destination)][0]
                del dict_infos_rem_pairs[(source, destination)]
                del dict_rem_ind_pairs[ind_pair]
            else:
                path = self.path_selector.return_path(subg_s_d, source, destination)
                self._update_iteration_num(increment_num_it = True, increment_cons_tries = False)
                if path is None:
                    self._update_iteration_num(increment_num_it = False, increment_cons_tries = True)
                    continue
                flow_amount = min(min(self.mfd_instance.aggregated_flow[path[j]][path[j+1]] for j in range(len(path)-1)),
                                    dict_infos_rem_pairs[(source, destination)][3],
                                    self.maximal_flow_amount)
                dict_infos_rem_pairs[(source, destination)][1] = path
                dict_infos_rem_pairs[(source, destination)][2] = flow_amount
                return [(source, destination)]
        # Return an empty list if no pair is found
        return []
    

    def select_path_for_each_pair(self, dict_infos_rem_pairs, dict_rem_ind_pairs):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair, 
        # capacity of the path, remaining_flow_value of the pair), ...}  
        # A 'dict' which has as key the index the pairs in 'self.pairs' and as value the corrsponding pair.
        # {indice of the pair in 'self.pairs' : the associated pair , ...}
        possible_pairs = []
        for rema_pair in list(dict_infos_rem_pairs.keys()):
            # Stop to iterate if the maximum number of iteration is attained
            if self._has_iterated_too_much (): break
            # Return a candidate source/destination pair (following the selected criteria)
            source, destination = rema_pair
            # Create a subgraphe containig the arcs which lies in a path from 'source' to 'destination'
            subg_s_d = self.subg_constructor.subgraph_source_dest(source, 
                                                                destination)
            # Return the candidate pair if there is a path linking the source to the destination and remove the pair otherwise 
            if not self.subg_constructor._exist_path ():
                ind_pair = dict_infos_rem_pairs[(source, destination)][0]
                del dict_infos_rem_pairs[(source, destination)]
                del dict_rem_ind_pairs[ind_pair]
            else:
                path = self.path_selector.return_path(subg_s_d, source, destination)
                self._update_iteration_num(increment_num_it = True, increment_cons_tries = False)
                if path is None:
                    self._update_iteration_num(increment_num_it = False, increment_cons_tries = True)
                    continue
                flow_amount = min(min(self.mfd_instance.aggregated_flow[path[j]][path[j+1]] for j in range(len(path)-1)),
                                    dict_infos_rem_pairs[(source, destination)][3],
                                    self.maximal_flow_amount)
                dict_infos_rem_pairs[(source, destination)][1] = path
                dict_infos_rem_pairs[(source, destination)][2] = flow_amount
                possible_pairs.append(rema_pair)
        
        # Return the paths along with their flow amounts and their associated pair id
        return self.pair_reorderer.reorder_pairs(possible_pairs, dict_infos_rem_pairs) if self.pair_reorderer is not None else possible_pairs
    

    def select_paths_pairs(self, pair_criteria, path_card_criteria, multi_flow, 
                           dict_infos_rem_pairs, dict_rem_ind_pairs):
        """
        Select a path for at least some of the remaining O/D pairs with possibly a filtering criteria
        The function returns two data structures :
            - A 'dict' which has as key the O/D pairs and as value a tuple of their indices in 'self.pairs', their chosen path and its capacity.
            {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair, 
                                    capacity of the path, remaining_flow_value of the pair), ...}
            
            - A 'dict' which has as key the index the pairs in 'self.pairs' and as value the corrsponding pair.
            {indice of the pair in 'self.pairs' : the associated pair , ...}
        """
        if pair_criteria == "max_remaining_flow_val": # The cases where only local criterias are used to select pairs
            # Construct the reodered pairs
            if path_card_criteria == "one_only":
                candidate_pairs = self.select_path_pair_max_rem_flow (dict_infos_rem_pairs, dict_rem_ind_pairs)
            
            elif path_card_criteria == "one_for_each":
                candidate_pairs = self.select_path_for_each_pair(dict_infos_rem_pairs, dict_rem_ind_pairs)

            chosen_pairs = self.path_filterer.filter_paths(candidate_pairs, 
                                                           self.mfd_instance.aggregated_flow, 
                                                           dict_infos_rem_pairs)
            #print(len(chosen_pairs))
            return chosen_pairs


    def desagregate_multi_flow (self, pair_criteria, path_card_criteria,
                                keep_logs = False, seed = 42, 
                                dir_save_name = None, show = False):
        # Test if the 'pair_criteria' not in 'PAIRS_CRITERIAS' and rpath_card_criteria' not in 'PATH_CARDINALITY_CRITERIAS' 
        if pair_criteria not in PAIRS_CRITERIAS or path_card_criteria not in PATH_CARDINALITY_CRITERIAS:
            print("The pair criteria 'pairs_criterias'.")
            sys.exit()
        # If 'path_card_criteria' is 'one_only' no need to reorder the pairs
        if self.pair_reorderer is not None and path_card_criteria == "one_only":
            print("Number of paths to be chosen does not correspond to the ordering criteria.")
            sys.exit()
        
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair,
        #                          capacity of the path, remaining_flow_value of the pair), ...}
        # {indice of the pair in 'self.pairs' : the associated pair , ...}
        # Create a list of matrices (which will contain the multiflow to be constructed)
        multi_flow = [init_graph_arc_attribute_vals(self.mfd_instance.adj_mat) 
                                                for _ in range(len(self.mfd_instance.pairs))]
        dict_infos_rem_pairs_paths = {pair:[ind, None, -1, self.mfd_instance.original_flow_values[ind]] for ind, pair in enumerate(self.mfd_instance.pairs)} 
        dict_rem_ind_pairs = {ind:pair for ind, pair in enumerate(self.mfd_instance.pairs)}
        self._init_iteration_num()
        
         # Select a  paths for one (or each) pair
        selected_pairs = self.select_paths_pairs(pair_criteria, path_card_criteria, multi_flow, dict_infos_rem_pairs_paths, dict_rem_ind_pairs)
        # Main loop
        while len(dict_infos_rem_pairs_paths) > 0 and not self._has_iterated_too_much ():
            # print(len(dict_infos_rem_pairs_paths))
            # Update the flow on each path
            self.update_multi_flow_network(multi_flow, selected_pairs, dict_infos_rem_pairs_paths)
            # Add the flow on the paths on the constructed transition matrix
            if self.constructed_transition_function is not None: 
                self.update_constructed_transition_function(selected_pairs, dict_infos_rem_pairs_paths)
            # Update the remaning pairs
            self.mfd_instance.update_remaining_pairs(dict_infos_rem_pairs_paths, dict_rem_ind_pairs)
            # Select new paths
            selected_pairs = self.select_paths_pairs(pair_criteria, path_card_criteria, multi_flow, dict_infos_rem_pairs_paths, dict_rem_ind_pairs)
            # Update the number of iteration
            self._update_iteration_num(increment_num_it = False, 
                                       increment_cons_tries = False,
                                       reset_cons_tries = (len(selected_pairs) > 0))
        print("Number of iteration ", self.num_it)
        if self._has_iterated_too_much (): print("!!!! Maximum iteration number attained. !!!!!")
        return multi_flow, self.generated_flow_values



if __name__ == "__main__":
    pass