import sys
import random
from copy import deepcopy
from abc import ABC, abstractmethod
import os
sys.path.append(os.getcwd())
from msmd.path_selectors import PATH_SELECTOR_TYPES, PathSelectorTransFunc, LargestFlowPathSelector
from msmd.multi_flow_desag_solver_utils import (PAIRS_CRITERIAS,
                                 PATH_CARDINALITY_CRITERIAS)
from msmd.multi_flow_desag_general_solver import MultiFlowDesagSolver
from msmd.subgraph_constructors import SubGraphConstructorTransF
from msmd.multi_flow_desag_instance_utils import init_support_transition_matrices, init_random_transition_matrices

###########################################################################################################################################################  
### A Multi Flow Desaggregation Class based which uses a transition function to selects paths
###########################################################################################################################################################


class MultiFlowDesagSolverTransF(MultiFlowDesagSolver):

    def __init__(self, 
                 mfd_instance,
                 max_path_length = 1000, 
                 total_nb_iterations = 1000,
                 max_nb_tries_find_path = 5,
                 maximal_flow_amount = float("inf"),
                 reodering_pairs_policy_name = None,
                 path_selector_type = "min_time_based",
                 construct_trans_function = False,
                 max_trans_func_successor = False,
                 exclude_chosen_nodes = False,
                 ignore_conflicts = False,
                 graph_representation = "adjacency_matrix"):
        # True iif we want to take the action associated to the maximum value in the transition function
        # Delete after creation
        self.path_desag_params = {"max_trans_func_successor":max_trans_func_successor}
        # Call parent constructor
        super().__init__(mfd_instance,
                        max_path_length = max_path_length, 
                        total_nb_iterations = total_nb_iterations,
                        max_nb_tries_find_path = max_nb_tries_find_path,
                        maximal_flow_amount = maximal_flow_amount,
                        reodering_pairs_policy_name = reodering_pairs_policy_name,
                        path_selector_type = path_selector_type,
                        construct_trans_function = construct_trans_function,
                        exclude_chosen_nodes = exclude_chosen_nodes,
                        ignore_conflicts = ignore_conflicts,
                        graph_representation = graph_representation)
        del self.path_desag_params


    def create_desaggregator_paths_attributes(self, path_selector_type, max_path_length, exclude_chosen_nodes):
        # Creation of a path selector
        if path_selector_type is None:
            self.path_selector = None
        
        elif path_selector_type == "trans_func_based" or path_selector_type == "trans_func_support" or path_selector_type == "random":
            self.path_selector = PathSelectorTransFunc(path_selector_type, max_path_length, 
                                                       exclude_chosen_nodes, self.mfd_instance)
            
            if path_selector_type == "trans_func_support" or path_selector_type == "random": 
                self.mfd_instance.update_transition_functions = False
            
            if path_selector_type == "random":
                init_random_transition_matrices(self.mfd_instance)

            if path_selector_type == "trans_func_support":
                init_support_transition_matrices(self.mfd_instance)       

        elif path_selector_type ==  "largest_flow_successor":
            self.path_selector = LargestFlowPathSelector(path_selector_type, max_path_length, 
                                                         exclude_chosen_nodes, self.mfd_instance.aggregated_flow)
            self.mfd_instance.update_transition_functions = False
            init_random_transition_matrices(self.mfd_instance)
        
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
        self.subg_constructor = SubGraphConstructorTransF(path_selector_type, 
                                                          self.mfd_instance, 
                                                          max_trans_func_successor = self.path_desag_params["max_trans_func_successor"],
                                                          graph_representation = self.graph_representation)
    


if __name__ == "__main__":
    pass