import sys
import numpy as np
import random
from abc import ABC, abstractmethod
from copy import deepcopy
import os
sys.path.append(os.getcwd())
from utils.graph_utils import successors, has_arc, init_graph_arc_attribute_vals
from msmd.stateless_RL_agents import (POLICY_BASED_TYPE_AGENTS,
                                            VALUE_BASED_TYPE_AGENTS,
                                            MIXED_TYPE_AGENTS)



# Constants
SUCCESSORS_SELECTOR_TYPES = {"standard", "exponential_decay"}


##########
#####################################  SUCCESSORS SELECTORS  #####################################
##########
class SuccessorSelector(ABC):

    @abstractmethod
    def chose_successor(self):
        pass



class RandomFuncSuccessorSelector(SuccessorSelector):

    def chose_successor(self, cur_node, graph_mat, cur_path = None):
        """
        Return a random successor of node 'cur_node' 
        """
        successor_nodes = successors(graph_mat, cur_node)
        if cur_path is not None:
            successor_nodes = [succ_node for succ_node in successor_nodes if succ_node not in cur_path]
        if len(successor_nodes) == 0: return None
        chosen_node = random.choice(successor_nodes)
        return chosen_node
    


class LargestFlowSuccessorSelector(SuccessorSelector):

    def chose_successor(self, cur_node, graph_mat, aggregated_flow, cur_path = None):
        """
        Return a random successor of node 'cur_node' 
        """
        successor_nodes = successors(graph_mat, cur_node)
        if cur_path is None:
            max_agg_flow = max(aggregated_flow[cur_node][succ_node] for succ_node in successor_nodes 
                                                                        if aggregated_flow[cur_node][succ_node] > 0)
            successor_nodes = [succ_node for succ_node in successor_nodes if aggregated_flow[cur_node][succ_node] == max_agg_flow]
        else:
            max_agg_flow = max(aggregated_flow[cur_node][succ_node] for succ_node in successor_nodes 
                                                                        if aggregated_flow[cur_node][succ_node] > 0 and\
                                                                            succ_node not in cur_path)
            successor_nodes = [succ_node for succ_node in successor_nodes if aggregated_flow[cur_node][succ_node] == max_agg_flow]
        if len(successor_nodes) == 0: return None
        chosen_node = random.choice(successor_nodes)
        return chosen_node



class TransFuncSuccessorSelector(SuccessorSelector):

    def chose_successor(self, path_selector, source, arc, graph_mat, cur_path = None):
        """
        Return successor arc using the transition function 
        """
        if arc is None:
            trans_func = path_selector.mfd_instance.transition_from_sources
            successor_arcs = [succ_arc for succ_arc in trans_func[source] if has_arc(graph_mat, succ_arc[0], succ_arc[1]) and\
                                                                                    trans_func[source][succ_arc] > 0]
            chosen_arc = random.choices(
                                successor_arcs,
                                weights = [trans_func[source][succ_arc] for succ_arc in successor_arcs],
                                k = 1)[0]
        else:
            trans_func = path_selector.mfd_instance.transition_function
            if cur_path is None:
                successor_arcs = [succ_arc for succ_arc in trans_func[arc] if has_arc(graph_mat, succ_arc[0], succ_arc[1]) and\
                                                                                trans_func[arc][succ_arc] > 0]
            else:
                successor_arcs = [succ_arc for succ_arc in trans_func[arc] if has_arc(graph_mat, succ_arc[0], succ_arc[1]) and\
                                                                                trans_func[arc][succ_arc] > 0 and\
                                                                                    succ_arc[1] not in cur_path]
            if len(successor_arcs) == 0: return None
            chosen_arc = random.choices(
                                successor_arcs,
                                weights = [trans_func[arc][succ_arc] for succ_arc in successor_arcs],
                                k = 1)[0]
        return chosen_arc



class RLSuccessorSelector(SuccessorSelector):

    def replace_agent_policy(self, rl_agent, policy, action_subspace):
        rl_agent.actions = action_subspace
        rl_agent.policy = [policy[rl_agent.actions_to_inds[action]] for action in action_subspace]
        sum_proba_actionsubspace = sum(rl_agent.policy)
        rl_agent.policy = np.array([p/sum_proba_actionsubspace for p in rl_agent.policy])


    def replace_agent_estimates(self, rl_agent, actions_estimates, action_subspace):
        rl_agent.actions = action_subspace
        rl_agent.actions_estimates = [actions_estimates[rl_agent.actions_to_inds[action]] for action in action_subspace]


    def chose_action_actionsubspace(self, rl_agent, action_subspace, ag_type):
        if ag_type == "Hier_cont_pursuit":
            print("Hierachical pusuite algorithm is not supported.")
            sys.exit()

        if ag_type in POLICY_BASED_TYPE_AGENTS or ag_type in MIXED_TYPE_AGENTS:
            # Replace total action space and policy with subaction space given in entry and its corresponing subpolicy
            actions, policy = rl_agent.actions, rl_agent.policy
            # Replace agent's policy
            self.replace_agent_policy(rl_agent, policy, action_subspace)
            # Chose action using subpolicy
            ind_action_acsubspace = rl_agent.chose_action()
            # Replace original action space and policy
            rl_agent.actions, rl_agent.policy = actions, policy # !!!!! (quik hack) peut etre qua ça va changer !!!!

        elif ag_type in VALUE_BASED_TYPE_AGENTS:
            # Replace total action space and estimates with subaction space given in entry and their corresponing estimates
            actions, actions_estimates = rl_agent.actions, rl_agent.actions_estimates
            # Replace agent's estimates
            self.replace_agent_estimates(rl_agent, actions_estimates, action_subspace)
            # Chose action using actions estimates of the actions in the subaction space
            ind_action_acsubspace = rl_agent.chose_action()
            # Replace original action space and policy
            rl_agent.actions, rl_agent.actions_estimates = actions, actions_estimates # !!!!! (quik hack) peut etre qua ça va changer !!!!
        
        else:
            print("Agent type not recognized.")
            sys.exit()

        return ind_action_acsubspace
    

    def chose_successor(self, path_selector, pair, elem, graph_mat, cur_path = None):
        """
        Return successor arc using the transition function 
        """
        if elem is None:
            trans_func, source, destination = path_selector.mfd_instance.transition_from_sources, pair[0], pair[1]
            successors = [succ_arc[1] for succ_arc in trans_func[source] if has_arc(graph_mat, succ_arc[0], succ_arc[1]) and\
                                                                            trans_func[source][succ_arc] > 0]
            id_ac_subspace = self.chose_action_actionsubspace (
                                        rl_agent = path_selector.source_agents[(source, destination)], 
                                        action_subspace = successors, 
                                        ag_type = path_selector.dict_parameters["ag_type"])
        else:
            trans_func, cur_node, source, destination = path_selector.mfd_instance.transition_function, elem[1], pair[0], pair[1]
            if cur_path is None:
                successors = [succ_arc[1] for succ_arc in trans_func[elem] if has_arc(graph_mat, succ_arc[0], succ_arc[1]) and\
                                                                                trans_func[elem][succ_arc] > 0]
            else:
                successors = [succ_arc[1] for succ_arc in trans_func[elem] if has_arc(graph_mat, succ_arc[0], succ_arc[1]) and\
                                                                                trans_func[elem][succ_arc] > 0 and\
                                                                                    succ_arc[1] not in cur_path]
            if len(successors) == 0: return None
            agent_key = (cur_node,) if path_selector.path_selector_type == "rl_node_based" else elem if path_selector.path_selector_type == "rl_arc_based" else None
            id_ac_subspace = self.chose_action_actionsubspace (
                                        rl_agent = path_selector.agents[(source, destination)+agent_key], 
                                        action_subspace = successors, 
                                        ag_type = path_selector.dict_parameters["ag_type"])
        
        return successors[id_ac_subspace]



class RLSuccessorSelectorExpoDecay(RLSuccessorSelector):
    
    def __init__ (self, graph, penalty_init_val, decay_param):
        self.original_penalty_mat = init_graph_arc_attribute_vals(graph, init_val = penalty_init_val)
        self.penalty_mat = deepcopy(self.original_penalty_mat)
        self.decay_param = decay_param
        self.cur_node = None


    def reset_wreproc (self):
        self.penalty_mat = deepcopy(self.original_penalty_mat)


    def update_penalty (self, u, v):
        self.penalty_mat[u][v] += 1


    def reprocess_weights(self, cur_node, successors, successors_weights, normalised = True):
        reproc_weights = [(self.decay_param**self.penalty_mat[cur_node][successors[ind_succ]])*successors_weights[ind_succ] for ind_succ in range(len(successors))]
        if normalised:
            sum_weights = sum(reproc_weights)
            return [weight/sum_weights for weight in reproc_weights]
        else:
            return reproc_weights
    

    def replace_agent_policy(self, rl_agent, policy, action_subspace):
        super().replace_agent_policy(rl_agent, policy, action_subspace)
        rl_agent.policy = self.reprocess_weights(self.cur_node, action_subspace, rl_agent.policy)


    def replace_agent_estimates(self, rl_agent, actions_estimates, action_subspace):
        super().replace_agent_estimates(rl_agent, actions_estimates, action_subspace)
        rl_agent.actions_estimates = self.reprocess_weights(self.cur_node, 
                                                            action_subspace, 
                                                            rl_agent.actions_estimates, 
                                                            normalised = False)
    

    def chose_successor(self, path_selector, pair, elem, graph_mat, cur_path = None):
        """
        Return successor arc using the transition function 
        """
        if elem is None:
            self.reset_wreproc ()
            trans_func, self.cur_node, destination = path_selector.mfd_instance.transition_from_sources, pair[0], pair[1]
            successors = [succ_arc[1] for succ_arc in trans_func[self.cur_node] if has_arc(graph_mat, succ_arc[0], succ_arc[1]) and\
                                                                            trans_func[self.cur_node][succ_arc] > 0]
            id_ac_subspace = self.chose_action_actionsubspace (
                                        rl_agent = path_selector.source_agents[(self.cur_node, destination)],
                                        action_subspace = successors, 
                                        ag_type = path_selector.dict_parameters["ag_type"])
            next_node = successors[id_ac_subspace]
        else:
            trans_func, self.cur_node, source, destination = path_selector.mfd_instance.transition_function, elem[1], pair[0], pair[1]
            if cur_path is None:
                successors = [succ_arc[1] for succ_arc in trans_func[elem] if has_arc(graph_mat, succ_arc[0], succ_arc[1]) and\
                                                                                trans_func[elem][succ_arc] > 0]
            else:
                successors = [succ_arc[1] for succ_arc in trans_func[elem] if has_arc(graph_mat, succ_arc[0], succ_arc[1]) and\
                                                                                trans_func[elem][succ_arc] > 0 and\
                                                                                    succ_arc[1] not in cur_path]
            if len(successors) == 0: return None
            agent_key = (self.cur_node,) if path_selector.path_selector_type == "rl_node_based" else elem if path_selector.path_selector_type == "rl_arc_based" else None
            id_ac_subspace = self.chose_action_actionsubspace (
                                        rl_agent = path_selector.agents[(source, destination)+agent_key], 
                                        action_subspace = successors, 
                                        ag_type = path_selector.dict_parameters["ag_type"])
            next_node = successors[id_ac_subspace]
        self.update_penalty (self.cur_node, next_node)
        return next_node