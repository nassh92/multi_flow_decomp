import sys
import numpy as np
import random
from abc import ABC, abstractmethod
from copy import deepcopy
import os
sys.path.append(os.getcwd())
from utils.graph_utils import make_path_simple
from msmd.stateless_RL_agents import (POLICY_BASED_TYPE_AGENTS,
                                            VALUE_BASED_TYPE_AGENTS,
                                            MIXED_TYPE_AGENTS,
                                            return_agent)
from msmd.successor_selectors import(LargestFlowSuccessorSelector,
                                           RandomFuncSuccessorSelector,
                                           TransFuncSuccessorSelector,
                                           RLSuccessorSelector,
                                           RLSuccessorSelectorExpoDecay,
                                           SUCCESSORS_SELECTOR_TYPES)



# Constants
PATH_SELECTOR_TYPES = {None, 
                       "min_time_based", "max_capacity_based", 
                       "random", "trans_func_based", 
                       "trans_func_support", "largest_flow_successor",
                       "rl_node_based", "rl_arc_based"}

RLDATA_INITIALIZATION_TYPES = {"uniform", "flow_based"}



##########
#####################################  PATH SELECTION - Path Selectors  #####################################
##########
class PathSelector(ABC):
    def __init__(self, path_selector_type, max_path_length, exclude_chosen_nodes):
        # Set path_selector_type after checking
        if path_selector_type not in PATH_SELECTOR_TYPES:
            print("Path selector type is unrecognized : ", path_selector_type)
            sys.exit()
        # The type of the path selector
        self.path_selector_type = path_selector_type
        # Set the maximum length of path (useful to not have a infinite loop during path construction)
        self.max_path_length = max_path_length
        # Set to True iif we want to exlude the already chosen nodes during path construction
        self.exclude_chosen_nodes = exclude_chosen_nodes


    @abstractmethod
    def return_path(self, graph_mat, source, destination):
        pass



class RandomPathSelector(PathSelector):

    def __init__(self, path_selector_type, max_path_length, exclude_chosen_nodes):
        super().__init__(path_selector_type, max_path_length, exclude_chosen_nodes)
        self.successor_selector = RandomFuncSuccessorSelector()


    def return_path (self, graph_mat, source, destination):
        """
        Return a random path of 'graph_mat' using the transition function
        """
        node = self.successor_selector.chose_successor(source, graph_mat)
        if node is None: return None
        path = [source, node]
        cur_path = None if not self.exclude_chosen_nodes else path
        path_length = 0
        while node != destination and path_length < self.max_path_length:
            node = self.successor_selector.chose_successor(node, graph_mat, cur_path = cur_path)
            if node is not None:
                path.append(node)
                path_length += 1
            else:
                #print("No successor in path.")
                return None
        
        if path_length == self.max_path_length and node != destination:
            print("Max path length is attained ", path_length)
            return None
        
        return make_path_simple (path)
    



class LargestFlowPathSelector(PathSelector):

    def __init__(self, path_selector_type, max_path_length, exclude_chosen_nodes, aggregated_flow):
        super().__init__(path_selector_type, max_path_length, exclude_chosen_nodes)
        self.aggregated_flow = aggregated_flow
        self.successor_selector = LargestFlowSuccessorSelector()


    def return_path (self, graph_mat, source, destination):
        """
        Return a random path of 'graph_mat' using the transition function
        """
        node = self.successor_selector.chose_successor(source, graph_mat, self.aggregated_flow)
        if node is None: return None
        path = [source, node]
        cur_path = None if not self.exclude_chosen_nodes else path
        path_length = 0
        while node != destination and path_length < self.max_path_length:
            node = self.successor_selector.chose_successor(node, graph_mat, self.aggregated_flow, cur_path = cur_path)
            if node is not None:
                path.append(node)
                path_length += 1
            else:
                #print("No successor in path.")
                return None
        
        if path_length == self.max_path_length and node != destination:
            print("Max path length is attained ", path_length)
            return None
        
        return make_path_simple (path)
    


class PathSelectorTransFunc(PathSelector):

    def __init__(self, path_selector_type, max_path_length, exclude_chosen_nodes, mfd_instance):
        super().__init__(path_selector_type, max_path_length, exclude_chosen_nodes)
        self.mfd_instance = mfd_instance
        self.successor_selector = TransFuncSuccessorSelector()


    def return_path (self, graph_mat, source, destination):
        """
        Return a random path of 'graph_mat' using the transition function
        """
        arc = self.successor_selector.chose_successor(self, source, None, graph_mat)
        if arc is None: return None
        path = [arc[0], arc[1]]
        cur_path = None if not self.exclude_chosen_nodes else path
        path_length = 0
        while arc[1] != destination and path_length < self.max_path_length:
            arc = self.successor_selector.chose_successor(self, source, arc, graph_mat, cur_path = cur_path)
            if arc is not None:
                path.append(arc[1])
                path_length += 1
            else:
                return None
        
        if path_length == self.max_path_length and arc[1] != destination:
            print("Max path length is attained ", path_length)
            return None
        
        return make_path_simple (path)



##########
#####################################  PATH SELECTION - RL based  #####################################
##########

class RLPathSelecAgentsHandler():

    def __init__(self, rl_data_init_type = "flow_based"):
        self.rl_data_init_type = rl_data_init_type
    

    def create_agents(self, rl_path_selector, dict_parameters, ac_est_normalised = False):
        adjacency_matrix = rl_path_selector.mfd_instance.adj_mat
        for source, destination in rl_path_selector.mfd_instance.pairs:
            actions = self.return_actionspace_agent (source, adjacency_matrix)
            initial_policy, initial_actions_estimates = self.return_initial_policy(
                                                                                   rl_path_selector,
                                                                                   source,
                                                                                   actions,
                                                                                   dict_parameters["ag_type"], 
                                                                                   ac_est_normalised = ac_est_normalised)
            if initial_actions_estimates is not None: dict_parameters["opt_params"]["initial_actions_estimates"] = initial_actions_estimates
            ag = return_agent(ag_type = dict_parameters["ag_type"], 
                                actions = actions,
                                initial_policy = initial_policy, 
                                lr = dict_parameters["lr"], 
                                eps = dict_parameters["eps"], 
                                opt_params = dict_parameters["opt_params"])
            rl_path_selector.source_agents[(source, destination)] = ag


    def return_actionspace_agent (self, node, adjacency_matrix):
        return [next_node for next_node in range(len(adjacency_matrix)) if adjacency_matrix[node][next_node] == 1]
    

    def return_initial_policy(self, rl_path_selector, agent_key, action_space, ag_type, ac_est_normalised = False):
        # !!!!!!! Revoir les questions sur l'initialisation !!!!!!!!!!
        initial_policy, initial_actions_estimates = None, None
        if ag_type in POLICY_BASED_TYPE_AGENTS:
            initial_policy = [init_data_val if (init_data_val := self.access_initalization_data(rl_path_selector, 
                                                                                                agent_key, 
                                                                                                succ_node)) != 0 else 1 
                                                for succ_node in action_space]
            sum_proba = sum(initial_policy) 
            initial_policy = np.array([proba/sum_proba for proba in initial_policy])
        
        elif ag_type in VALUE_BASED_TYPE_AGENTS:
            initial_actions_estimates = [self.access_initalization_data(rl_path_selector, agent_key, succ_node) for succ_node in action_space]
            if ac_est_normalised:
                sum_est_ac = sum(initial_actions_estimates)
                initial_actions_estimates = [ac_est/sum_est_ac for ac_est in initial_actions_estimates] if sum_est_ac != 0 else [0]*len(initial_actions_estimates)

        elif ag_type in MIXED_TYPE_AGENTS:
            initial_policy = [init_data_val if (init_data_val := self.access_initalization_data(rl_path_selector, 
                                                                                                agent_key, 
                                                                                                succ_node)) != 0 else 1 
                                                for succ_node in action_space]
            sum_proba = sum(initial_policy)
            initial_policy = np.array([proba/sum_proba for proba in initial_policy])
            initial_actions_estimates = [self.access_initalization_data(rl_path_selector, agent_key, succ_node) for succ_node in action_space]
            if ac_est_normalised:
                sum_est_ac = sum(initial_actions_estimates)
                initial_actions_estimates = [ac_est/sum_est_ac for ac_est in initial_actions_estimates] if sum_est_ac != 0 else [0]*len(initial_actions_estimates)
        return initial_policy, initial_actions_estimates
    

    def access_initalization_data(self):
        return 1
    


class RLNodePathSelecAgentsHandler(RLPathSelecAgentsHandler):
    
    def create_agents(self, rl_path_selector, dict_parameters, ac_est_normalised = False):
        super().create_agents(rl_path_selector, dict_parameters, ac_est_normalised = ac_est_normalised)
        adjacency_matrix = rl_path_selector.mfd_instance.adj_mat
        for source, destination in rl_path_selector.mfd_instance.pairs:
            for node in range(len(adjacency_matrix)):
                actions = self.return_actionspace_agent (node, adjacency_matrix)
                if len(actions) > 0:
                    initial_policy, initial_actions_estimates = self.return_initial_policy(
                                                                                        rl_path_selector, 
                                                                                        node, 
                                                                                        actions, 
                                                                                        dict_parameters["ag_type"], 
                                                                                        ac_est_normalised = ac_est_normalised)
                    if initial_actions_estimates is not None: dict_parameters["opt_params"]["initial_actions_estimates"] = initial_actions_estimates
                    ag = return_agent(ag_type = dict_parameters["ag_type"], 
                                        actions = actions,
                                        initial_policy = initial_policy, 
                                        lr = dict_parameters["lr"], 
                                        eps = dict_parameters["eps"], 
                                        opt_params = dict_parameters["opt_params"])
                    rl_path_selector.agents[(source, destination, node)] = ag
    

    def access_initalization_data(self, rl_path_selector, agent_key, succ_node):
        if self.rl_data_init_type == "uniform": return super().access_initalization_data()
        elif self.rl_data_init_type == "flow_based":
            return rl_path_selector.mfd_instance.aggregated_flow[agent_key][succ_node]



class RLArcPathSelecAgentsHandler(RLPathSelecAgentsHandler):
    
    def create_agents(self, rl_path_selector, dict_parameters, ac_est_normalised = False):
        super().create_agents(rl_path_selector, dict_parameters, ac_est_normalised = ac_est_normalised)
        adjacency_matrix = rl_path_selector.mfd_instance.adj_mat
        for source, destination in rl_path_selector.mfd_instance.pairs:
            for node in range(len(adjacency_matrix)):
                for succ_node in range(len(adjacency_matrix)):
                    if adjacency_matrix[node][succ_node] == 1:
                        actions = self.return_actionspace_agent (succ_node, adjacency_matrix)
                        if len(actions) > 0:
                            initial_policy, initial_actions_estimates = self.return_initial_policy(
                                                                                                rl_path_selector, 
                                                                                                (node, succ_node), 
                                                                                                actions, 
                                                                                                dict_parameters["ag_type"],
                                                                                                ac_est_normalised = ac_est_normalised)
                            if initial_actions_estimates is not None: dict_parameters["opt_params"]["initial_actions_estimates"] = initial_actions_estimates
                            ag = return_agent(ag_type = dict_parameters["ag_type"], 
                                                actions = actions,
                                                initial_policy = initial_policy, 
                                                lr = dict_parameters["lr"], 
                                                eps = dict_parameters["eps"], 
                                                opt_params = dict_parameters["opt_params"])
                            rl_path_selector.agents[(source, destination, node, succ_node)] = ag
    
    
    def access_initalization_data(self, rl_path_selector, agent_key, succ_node):
        if self.rl_data_init_type == "uniform": return super().access_initalization_data()
        elif self.rl_data_init_type == "flow_based":
            if isinstance(agent_key, tuple):
                succ_arc = (agent_key[1], succ_node)
                return rl_path_selector.mfd_instance.transition_function[agent_key][succ_arc] if succ_arc in rl_path_selector.mfd_instance.transition_function[agent_key] else sys.exit()
            else:
                succ_arc = (agent_key, succ_node)
                return rl_path_selector.mfd_instance.transition_from_sources[agent_key][succ_arc]


#####################################  RL Path Selector  #####################################
class RLPathSelector(PathSelector):
    
    def __init__(self, 
                 mfd_instance, dict_parameters, path_selector_type, max_path_length, 
                 ac_est_normalised = False, exclude_chosen_nodes = False, 
                 successor_selector_type = "standard", rl_data_init_type = "flow_based", 
                 opt_params = None):
        # Exit if 'successor_selector' is not recognized 
        if successor_selector_type not in SUCCESSORS_SELECTOR_TYPES or rl_data_init_type not in RLDATA_INITIALIZATION_TYPES:
            print("successor selector or 'rl_data_init_type' is not recognized")
            sys.exit()

        # Calling the constructor
        super().__init__(path_selector_type, max_path_length, exclude_chosen_nodes)

        # The mfd instance
        self.mfd_instance = mfd_instance

        # Intializing the parameters of the RL algorithm
        self.dict_parameters = dict_parameters

        # Initialise selected paths
        self.selected_paths = []

        # Initialization of the agents handler (help create the agents)
        self.agents, self.source_agents = {}, {}
        if path_selector_type == "rl_node_based":
            self.agents_handler = RLNodePathSelecAgentsHandler(rl_data_init_type)

        elif path_selector_type == "rl_arc_based":
            self.agents_handler = RLArcPathSelecAgentsHandler(rl_data_init_type)
        
        # Agents creations
        self.agents_handler.create_agents(self, dict_parameters, ac_est_normalised = ac_est_normalised)

        # Successor selection creations
        if successor_selector_type == "standard":
            self.successor_selector = RLSuccessorSelector()

        elif successor_selector_type == "exponential_decay":
            if opt_params is None or (opt_params is not None and ("penalty_init_val" not in opt_params or "decay_param" not in opt_params)):
                print("Penalty keys error in 'opt_params'.")
                sys.exit()
            self.successor_selector = RLSuccessorSelectorExpoDecay(
                                                                len(self.mfd_instance.adj_mat),
                                                                opt_params["penalty_init_val"],
                                                                opt_params["decay_param"])


    def reset_selected_paths(self):
        self.selected_paths = []
    

    def add_path(self, chosen_path):
        self.selected_paths.append(chosen_path)
    

    def return_path(self, subg_s_d, source, destination):
        """
        Methode abstraite implementant le choix d'une action suivant le type d'agent
        """
        node = self.successor_selector.chose_successor(self, (source, destination), None , subg_s_d)
        if node is None: return None
        path = [source, node]
        cur_path = None if not self.exclude_chosen_nodes else path
        path_length = 0
        while node != destination and path_length < self.max_path_length:
            node = self.successor_selector.chose_successor(self, (source, destination), (path[-2], path[-1]), subg_s_d, cur_path = cur_path)
            if node is not None:
                path.append(node)
                path_length += 1
            else:
                #print("No successors.")
                return None

        if path_length == self.max_path_length:
            print("Max path length is attained ", path_length)
            return None
        
        chosen_path = make_path_simple (path)
        self.add_path(chosen_path)
        return chosen_path  


    def update_agents_policies(self, reward):
        """
        Mise à jour de la politique
        """
        for path in self.selected_paths:
            source, destination = path[0], path[-1]
            rl_agent = self.source_agents[(source, destination)] 
            id_action = rl_agent.actions_to_inds[path[1]]
            rl_agent.update_policy(id_action, reward)

            for ind_node in range(1, len(path)-1):
                cur_arc = (path[ind_node-1], path[ind_node])
                agent_key = (cur_arc[1],) if self.path_selector_type == "rl_node_based" else cur_arc if self.path_selector_type == "rl_arc_based" else None
                rl_agent = self.agents[(source, destination)+agent_key]
                id_action = rl_agent.actions_to_inds[path[ind_node+1]]
                rl_agent.update_policy(id_action, reward)

    