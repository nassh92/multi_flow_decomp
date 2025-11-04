import sys
import random
from copy import deepcopy
import numpy as np
import os
sys.path.append(os.getcwd())
from msmd.stateless_RL_agents import POLICY_BASED_TYPE_AGENTS, VALUE_BASED_TYPE_AGENTS, MIXED_TYPE_AGENTS
from utils.graph_utils import path_intersection, check_arc_inclusion
from utils.random_utils import weighted_shuffle


##########
#####################################  CONSTANTS   #####################################
##########
PAIRS_CRITERIAS = {"max_remaining_flow_val"}


PATH_CARDINALITY_CRITERIAS = {"one_only", "one_for_each"}


STOCHASTIC_PAIRS_POLICIES_NAMES = {"uniform", "proba_remaining_flow_val",
                                   "uniform_rp", "proba_remaining_flow_val_rp"}


SEMI_STOCHASTIC_PAIRS_POLICIES_NAMES = {"remaining_max_flow"}



##########
#####################################  PATH REODERING   #####################################
##########
class WeightPairsPolicy:
    def __init__ (self, reodering_pairs_policy_name, original_flow_values):
            self.reodering_pairs_policy_name = reodering_pairs_policy_name
            self.init_pairs_weights(original_flow_values)

    def init_pairs_weights(self, original_flow_values):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair, 
        # capacity of the path, remaining_flow_value of the pair), ...} 
        if self.reodering_pairs_policy_name == "uniform" or self.reodering_pairs_policy_name == "uniform_rp":
            self.weights_pairs = [int(fl_val > 0) for fl_val in original_flow_values]
        elif self.reodering_pairs_policy_name == "proba_remaining_flow_val" or self.reodering_pairs_policy_name == "proba_remaining_flow_val_rp":
            self.weights_pairs = [fl_val for fl_val in original_flow_values]
    
    def process_pairs_weights(self, pairs, dict_infos_rem_pairs):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair, 
        # capacity of the path, remaining_flow_value of the pair), ...} 
        if self.reodering_pairs_policy_name == "uniform_rp": # Reprocess the weights
            self.weights_pairs = [int(dict_infos_rem_pairs[pair][3] > 0) for pair in pairs]
        elif self.reodering_pairs_policy_name == "proba_remaining_flow_val_rp": # Reprocess the weights
            self.weights_pairs = [dict_infos_rem_pairs[pair][3] for pair in pairs]

    def return_pairs_weights(self):
        return self.weights_pairs

    def reorder_pairs(self, possible_pairs, dict_infos_rem_pairs):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair, 
        # capacity of the path, remaining_flow_value of the pair), ...} 
        if len(possible_pairs) == 0: return possible_pairs
        self.process_pairs_weights()
        ls_all_weights = self.return_pairs_weights()
        ls_possible_weights = [ls_all_weights[dict_infos_rem_pairs[pair][0]] for pair in possible_pairs]
        reordered_pairs = weighted_shuffle(possible_pairs, 
                                            ls_possible_weights)
        return reordered_pairs


class SemiStochasticReoderer:
    def __init__ (self, reodering_pairs_policy_name):
            self.reodering_pairs_policy_name = reodering_pairs_policy_name
        
    def chose_candidate_pair(self, ls_infos_rem_flow):
        if self.reodering_pairs_policy_name == "remaining_max_flow":
            rem_max_flow = max(elem[1] for elem  in ls_infos_rem_flow)
            rem_max_flow_pairs = [pair for pair, fl_val in ls_infos_rem_flow if fl_val == rem_max_flow]
            return rem_max_flow_pairs[random.randint(0, len(rem_max_flow_pairs)-1)], rem_max_flow

    def reorder_pairs(self, possible_pairs, dict_infos_rem_pairs):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair, 
        # capacity of the path, remaining_flow_value of the pair), ...}
        if len(possible_pairs) == 0: return possible_pairs
        ls_infos_rem_flow = [(pair, dict_infos_rem_pairs[pair][3]) for pair in possible_pairs]
        reordered_pairs = []
        while len(ls_infos_rem_flow) > 0:
            pair_to_add, cur_max_flow = self.chose_candidate_pair(ls_infos_rem_flow)
            reordered_pairs.append(pair_to_add)
            ls_infos_rem_flow.remove((pair_to_add, cur_max_flow))
        return reordered_pairs



##########
#####################################  PATH FILTERING   #####################################
##########
class PathFilterer:
    def __init__(self, ignore_conflicts = False):
        self.ignore_conflicts = ignore_conflicts

    """
    def _process_conflicts_costs(self, 
                                 pair_add, pair_test,
                                 arcs_intersection, 
                                 dict_infos_rem_pairs,
                                 chosen_pairs,
                                 aggregated_flow,
                                 multi_flow):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair,
        #                          capacity of the path, remaining_flow_value of the pair), ...}
        # For each arc test if there is a conflict between 'path_to_add' and 'path_to_test' and update the costs accordingly
        cost_add, cost_test, conflict = 0, 0, False
        for arc in arcs_intersection: 
            # Process the Id pairs (id in self.pairs) for the !remaining! pairs 
            # for which 'arc' belongs to the current 'chosen_paths' ('chosen_id_pairs') or 'path1'
            # ids_valid_pair_arc = [id_pair for id_pair in ids_remaining_pairs if id_pair in chosen_id_pairs+[id_pair_p1] and\
            #                                                                        multi_flow[id_pair][arc[0]][arc[1]] > 0]
            valid_pairs_arc = {pair for pair in chosen_pairs if check_arc_inclusion(arc, dict_infos_rem_pairs[pair][1])} | {pair_add}
            sum_flow_arc_paths = sum(dict_infos_rem_pairs[pair][2] for pair in valid_pairs_arc)
            
            if sum_flow_arc_paths > aggregated_flow[arc[0]][arc[1]]: # aggregated_flow[arc[0]][arc[1]] > 0 ???
                conflict = True
                if sum_flow_arc_paths - dict_infos_rem_pairs[pair_test][2] <= aggregated_flow[arc[0]][arc[1]]:
                    cost_add += multi_flow[dict_infos_rem_pairs[pair_add][0]][arc[0]][arc[1]]
                    cost_test += multi_flow[dict_infos_rem_pairs[pair_test][0]][arc[0]][arc[1]]
                else: 
                    return float('inf'), 0, conflict
        return cost_add, cost_test, conflict
    """
    
    
    def _process_conflicts_costs(self, 
                                 pair_add, pair_test,
                                 arcs_intersection, 
                                 dict_infos_rem_pairs,
                                 chosen_pairs,
                                 aggregated_flow,
                                 multi_flow):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair,
        #                          capacity of the path, remaining_flow_value of the pair), ...}
        # For each arc test if there is a conflict between 'path_to_add' and 'path_to_test' and update the costs accordingly
        cost_add, cost_test = 0, 0
        for arc in arcs_intersection: 
            # Process the Id pairs (id in self.pairs) for the !remaining! pairs 
            # for which 'arc' belongs to the current 'chosen_paths' ('chosen_id_pairs') or 'path1'
            # ids_valid_pair_arc = [id_pair for id_pair in ids_remaining_pairs if id_pair in chosen_id_pairs+[id_pair_p1] and\
            #                                                                        multi_flow[id_pair][arc[0]][arc[1]] > 0]
            valid_pairs_arc = {pair for pair in chosen_pairs if check_arc_inclusion(arc, dict_infos_rem_pairs[pair][1])} | {pair_add}
            sum_flow_arc_paths = sum(dict_infos_rem_pairs[pair][3] for pair in valid_pairs_arc)
            if sum_flow_arc_paths > aggregated_flow[arc[0]][arc[1]]:
                if sum_flow_arc_paths - dict_infos_rem_pairs[pair_add][3] + dict_infos_rem_pairs[pair_add][3] <= aggregated_flow[arc[0]][arc[1]]:
                    print("HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                    sys.exit()
                    #cost_add += multi_flow[dict_infos_rem_pairs[pair_add][0]][arc[0]][arc[1]]
                    #cost_test += multi_flow[dict_infos_rem_pairs[pair_test][0]][arc[0]][arc[1]]
                else: 
                    return float('inf'), 0, True
        conflict = (cost_add != 0)
        return cost_add, cost_test, conflict
    


    def _is_path_added(self, 
                       pair_add,
                       arcs_intersection, 
                       dict_infos_rem_pairs,
                       chosen_pairs,
                       aggregated_flow):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair,
        #                          capacity of the path, remaining_flow_value of the pair), ...}
        # For each arc test if there is a conflict between 'path_to_add' and 'path_to_test' and update the costs accordingly
        for arc in arcs_intersection: 
            # Process the Id pairs (id in self.pairs) for the !remaining! pairs 
            # for which 'arc' belongs to the current 'chosen_paths' ('chosen_id_pairs') or 'path1'
            # ids_valid_pair_arc = [id_pair for id_pair in ids_remaining_pairs if id_pair in chosen_id_pairs+[id_pair_p1] and\
            #                                                                        multi_flow[id_pair][arc[0]][arc[1]] > 0]
            valid_pairs_arc = {pair for pair in chosen_pairs if check_arc_inclusion(arc, dict_infos_rem_pairs[pair][1])} | {pair_add}
            sum_rfl_val_pairs_arc = sum(dict_infos_rem_pairs[pair][3] for pair in valid_pairs_arc)
            if sum_rfl_val_pairs_arc > aggregated_flow[arc[0]][arc[1]]: return False
        return True
    

    def filter_paths(self, possible_pairs, aggregated_flow, dict_infos_rem_pairs):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair, 
        # capacity of the path, remaining_flow_value of the pair), ...} 
        # Program assumption : 'self.remaining_pairs' has at least one element
        # Initialize a list containing data on the remaining pairs along with their associated weights
        # [(a remaining pair, pair id in 'self.pairs', a path conneting the pair, flow to be added in the path), ...]
        if len(possible_pairs) == 0: return possible_pairs
        chosen_pairs = [possible_pairs[0]]
        for pair_to_add in possible_pairs[1:]:
            path_to_add = dict_infos_rem_pairs[pair_to_add][1]

            for pair_to_test in chosen_pairs:
                path_to_test = dict_infos_rem_pairs[pair_to_test][1]
                arcs_intersection = path_intersection(path_to_add, path_to_test)
                add_path = self._is_path_added(pair_to_add,
                                               arcs_intersection, 
                                               dict_infos_rem_pairs,
                                               chosen_pairs,
                                               aggregated_flow)
                if not add_path: break 
            
            if add_path: chosen_pairs.append(pair_to_add)
        
        return chosen_pairs
    

        """
        def filter_paths(self, possible_pairs, multi_flow, aggregated_flow, dict_infos_rem_pairs):
        # {pair in 'self.pairs' : (indice of the pair in 'self.pairs', the chosen path for the pair, 
        # capacity of the path, remaining_flow_value of the pair), ...} 
        # Program assumption : 'self.remaining_pairs' has at least one element
        # Initialize a list containing data on the remaining pairs along with their associated weights
        # [(a remaining pair, pair id in 'self.pairs', a path conneting the pair, flow to be added in the path), ...]
        if len(possible_pairs) == 0: return possible_pairs
        chosen_pairs = [possible_pairs[0]]
        for pair_to_add in possible_pairs[1:]:
            path_to_add, add_path = dict_infos_rem_pairs[pair_to_add][1], True
            pairs_to_remove = []

            for pair_to_test in chosen_pairs:
                path_to_test = dict_infos_rem_pairs[pair_to_test][1]
                arcs_intersection = path_intersection(path_to_add, path_to_test)
                cost_add, cost_test, conflict = self._process_conflicts_costs(pair_to_add, pair_to_test,
                                                                            arcs_intersection, 
                                                                            dict_infos_rem_pairs,
                                                                            chosen_pairs,
                                                                            aggregated_flow,
                                                                            multi_flow)
                
                if conflict: # if there is a conflict, judge from relative costs of the paths
                    if self.ignore_conflicts or\
                        (cost_test <= cost_add and not self.ignore_conflicts): # can't add 'path_to_add', so we don't remove any paths and we go to the next iteration
                        add_path = False
                        pairs_to_remove = []
                        break
                    else: # add path of index j into the path which will be deleted
                        pairs_to_remove.append(pair_to_test)    
            
            for pair_to_rem in pairs_to_remove:
                chosen_pairs.remove(pair_to_rem)
            
            if add_path: chosen_pairs.append(pair_to_add)
        
        return chosen_pairs
        """