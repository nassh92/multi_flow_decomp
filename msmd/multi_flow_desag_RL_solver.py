import sys
import random
import numpy as np
from copy import deepcopy
import os
sys.path.append(os.getcwd())
from utils.metrics import transition_function_residue, flow_val_residue, flow_residue
from utils.graph_utils import create_isolated_nodes_graph
from msmd.multi_flow_desag_general_solver import MultiFlowDesagSolver
from msmd.path_selectors import RLPathSelector, PATH_SELECTOR_TYPES
from msmd.subgraph_constructors import SubGraphConstructorRL
from msmd.multi_flow_desag_solver_utils import (PAIRS_CRITERIAS,
                                 PATH_CARDINALITY_CRITERIAS,
                                 STOCHASTIC_PAIRS_POLICIES_NAMES,
                                 SEMI_STOCHASTIC_PAIRS_POLICIES_NAMES,
                                 WeightPairsPolicy, 
                                 SemiStochasticReoderer) 


class MultiFlowDesagRLSolver(MultiFlowDesagSolver):

    def __init__(self, 
                 mfd_instance,
                 path_selector_type,
                 dict_parameters,
                 max_path_length = 1000, 
                 max_nb_it_episode = 1000,
                 nb_episodes = 20,
                 max_nb_tries_find_path = 5,
                 maximal_flow_amount = float("inf"),
                 reodering_pairs_policy_name = None,
                 ac_est_normalised = False,
                 exclude_chosen_nodes = False,
                 successor_selector_type = "standard",
                 rl_data_init_type = "flow_based",
                 store_perfs_evol_path = None,
                 ignore_conflicts = False,
                 matrix_representation = True,
                 opt_params = None):
        # The parameters for path creation (only used for creation)
        self.path_desag_params = {"dict_parameters_rl":dict_parameters,
                                  "ac_est_normalised":ac_est_normalised,
                                  "successor_selector_type":successor_selector_type,
                                  "rl_data_init_type":rl_data_init_type,
                                  "opt_params":opt_params}  
        # Set maximum number of iterations per episodes
        self.max_nb_it_episode = max_nb_it_episode
        # Number of episodes
        self.nb_episodes = nb_episodes
        # True iif we want to save the evolution of the performance metrics
        self.store_perfs_evol_path = store_perfs_evol_path
        self.performance_metrics_evol = None if store_perfs_evol_path is None else []
        # Calling the parent constructor
        super().__init__(
                        mfd_instance,
                        max_path_length = max_path_length,
                        max_nb_tries_find_path = max_nb_tries_find_path,
                        maximal_flow_amount = maximal_flow_amount,
                        reodering_pairs_policy_name = reodering_pairs_policy_name,
                        path_selector_type = path_selector_type,
                        construct_trans_function = True,
                        exclude_chosen_nodes = exclude_chosen_nodes,
                        ignore_conflicts = ignore_conflicts,
                        matrix_representation = matrix_representation)
        del self.path_desag_params


    def create_desaggregator_paths_attributes(self, path_selector_type, max_path_length, exclude_chosen_nodes):
        # Creation of a path selector
        if  path_selector_type is None:
            print("Path selector must not be None.")
            sys.exit()

        elif path_selector_type not in PATH_SELECTOR_TYPES:
            print("Path selector type is unrecognized : ", path_selector_type)
            sys.exit()

        else:
            self.path_selector = RLPathSelector(self.mfd_instance, 
                                                self.path_desag_params["dict_parameters_rl"], 
                                                path_selector_type, max_path_length, 
                                                ac_est_normalised = self.path_desag_params["ac_est_normalised"],
                                                exclude_chosen_nodes = exclude_chosen_nodes,
                                                successor_selector_type = self.path_desag_params["successor_selector_type"],
                                                rl_data_init_type = self.path_desag_params["rl_data_init_type"],
                                                opt_params = self.path_desag_params["opt_params"])
            
        # Set a subgraph filterer
        self.subg_constructor = SubGraphConstructorRL(path_selector_type, 
                                                      self.mfd_instance, 
                                                      self)
    

    def init_episode(self):
        # Reset the instance
        self.mfd_instance.reset_instance()
        
        # Reset path selector
        self.path_selector.reset_selected_paths()

        # Generate flow values reset
        self.generated_flow_values = [0]*len(self.mfd_instance.original_flow_values)

        # Set the newly formed transition function
        self.constructed_transition_function = {(u, v):{(v,w):0 for w in range(len(self.mfd_instance.adj_mat)) 
                                                                    if self.mfd_instance.adj_mat[v][w] == 1}
                                                                        for u in range(len(self.mfd_instance.adj_mat)) 
                                                                            for v in range(len(self.mfd_instance.adj_mat)) 
                                                                                if self.mfd_instance.adj_mat[u][v] == 1}

        # Reset the pair reoderer used to reorder the pairs in case we select a path for each of the pairs
        if self.reodering_pairs_policy_name is None:
            self.pair_reorderer = None
        
        elif self.reodering_pairs_policy_name in STOCHASTIC_PAIRS_POLICIES_NAMES:
            self.pair_reorderer = WeightPairsPolicy(self.reodering_pairs_policy_name, self.mfd_instance.original_flow_values)
        
        elif self.reodering_pairs_policy_name in SEMI_STOCHASTIC_PAIRS_POLICIES_NAMES:
            self.pair_reorderer = SemiStochasticReoderer(self.reodering_pairs_policy_name)

        else:
            print("Reorder pair name is unrecognized.")
            sys.exit()

    
    def _has_iterated_too_much (self):
        return self.num_it >= self.max_nb_it_episode or self.nb_tries_find_path >= self.max_nb_tries_find_path
    
    
    def process_perfs(self, multi_flow, coeff1, coeff2, coeff3):
        fl_val_res, fl_res, transf_res = 0, 0, 0

        if coeff1 != 0:
            fl_val_res = flow_val_residue (self.generated_flow_values, 
                                           self.mfd_instance.original_flow_values)
            
        if coeff2 != 0:
            fl_res = flow_residue (multi_flow, 
                                   [[0 for v in range(len(self.mfd_instance.aggregated_flow))] 
                                            for u in range(len(self.mfd_instance.aggregated_flow))], 
                                   self.mfd_instance.original_aggregated_flow)
            
        if coeff3 != 0:
            transf_res = transition_function_residue (self.mfd_instance.original_transition_function, 
                                                      self.constructed_transition_function, 
                                                      self.mfd_instance.original_aggregated_flow)
        
        total_weight_error = coeff1*fl_val_res + coeff2*fl_res + coeff3*transf_res

        if total_weight_error > 1.0000001 or total_weight_error < 0:
            print("Total weigted error is out of range.")
            sys.exit()
        
        return 1 - total_weight_error, fl_val_res, fl_res, transf_res
    

    def store_performance_metrics(self, ls_perfs):
        # reward, fl_val_res, fl_res, transf_res
        self.performance_metrics_evol.append({"reward":ls_perfs[0], 
                                              "fl_val_res":ls_perfs[1], 
                                              "fl_res":ls_perfs[2], 
                                              "transf_res":ls_perfs[3]})
    
    
    def desagregate_multi_flow (self, pair_criteria, path_card_criteria,
                                coeff1, coeff2, coeff3,
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
        for _ in range(self.nb_episodes):
            # Initializations
            self.init_episode()
            self._init_iteration_num()
            multi_flow = [create_isolated_nodes_graph(len(self.mfd_instance.adj_mat), 
                                                      matrix_representation = self.matrix_representation) 
                                                            for _ in range(len(self.mfd_instance.pairs))]
            dict_infos_rem_pairs_paths = {pair:[ind, None, -1, self.mfd_instance.original_flow_values[ind]] for ind, pair in enumerate(self.mfd_instance.pairs)} 
            dict_rem_ind_pairs = {ind:pair for ind, pair in enumerate(self.mfd_instance.pairs)}
            
            # Select a  paths for one (or each) pair
            selected_pairs = self.select_paths_pairs(pair_criteria, path_card_criteria, multi_flow, dict_infos_rem_pairs_paths, dict_rem_ind_pairs)
            # Main Loop
            while len(dict_infos_rem_pairs_paths) > 0 and not self._has_iterated_too_much():
                # Update the flow on each path
                self.update_multi_flow_network(multi_flow, selected_pairs, dict_infos_rem_pairs_paths)
                # Add the flow on the paths on the constructed transition matrix
                self.update_constructed_transition_function(selected_pairs, dict_infos_rem_pairs_paths)
                # Update the remaning pairs
                self.mfd_instance.update_remaining_pairs(dict_infos_rem_pairs_paths, dict_rem_ind_pairs)
                # Select new paths
                selected_pairs = self.select_paths_pairs(pair_criteria, path_card_criteria, multi_flow, dict_infos_rem_pairs_paths, dict_rem_ind_pairs)
                # Update the number of iteration
                self._update_iteration_num(increment_num_it = False, 
                                           increment_cons_tries = False,
                                           reset_cons_tries = (len(selected_pairs) > 0))
            
            if self._has_iterated_too_much(): print("!!!! Maximum iteration number attained. !!! ")
            # Process relevant perfomance metric (including the reward)
            reward, fl_val_res, fl_res, transf_res = self.process_perfs(multi_flow, coeff1, coeff2, coeff3)
            # Save the relevant performance metrics if we want to
            if self.performance_metrics_evol is not None: self.store_performance_metrics([reward, fl_val_res, fl_res, transf_res])
            # Update the policies of the agent
            self.path_selector.update_agents_policies(reward)
        # Save the performances metrics
        if self.performance_metrics_evol is not None: np.save(self.store_perfs_evol_path, self.performance_metrics_evol)

        return multi_flow, self.generated_flow_values