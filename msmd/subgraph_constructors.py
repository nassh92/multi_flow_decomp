import sys
import random
from copy import deepcopy
import numpy as np
from abc import ABC, abstractmethod
import os
sys.path.append(os.getcwd())
from utils.graph_utils import has_arc, add_arc, create_isolated_nodes_graph, init_graph_arc_attribute_vals
from msmd.stateless_RL_agents import POLICY_BASED_TYPE_AGENTS, VALUE_BASED_TYPE_AGENTS, MIXED_TYPE_AGENTS
from utils.shortest_path_solvers import DijkstraShortestPathsSolver



##########
#####################################  Subgraph Constructor  #####################################
##########
class SubGraphConstructor(ABC):
    
    @abstractmethod
    def subgraph_source_dest(self):
        pass



##########
#####################################  Subgraph Constructor  #####################################
##########
class SubGraphBestPathsConstructor(SubGraphConstructor):

    def __init__(self, path_selector_type, mfd_instance, matrix_representation = True):
        # The path selector type
        self.path_selector_type = path_selector_type
        # The multi flow desaggregation instance
        self.mfd_instance = mfd_instance
        # Boolean showing how the graph is represented
        self.matrix_representation = matrix_representation
    
    
    def _exist_path (self):
        if self.dijkstra_solver.mode == "min_distance":
            return self.dijkstra_solver.path_estimates[self.destination] != float("inf")
        
        elif self.dijkstra_solver.mode == "max_capacity":
            return self.dijkstra_solver.path_estimates[self.destination] != 0


    def subgraph_source_dest(self, 
                             source, 
                             destination):
        self.source, self.destination = source, destination
        # Set the weight matrix
        if self.path_selector_type == "min_time_based":
            mode = "min_distance"
            weight_mat = self.mfd_instance.transport_times
         
        elif self.path_selector_type  == "max_capacity_based":
            mode = "max_capacity"
            weight_mat = self.mfd_instance.aggregated_flow

        else:
            print("Path selector is not reconized")
            sys.exit()
        
        # Create an instanceof the Dijkstra solver
        self.dijkstra_solver = DijkstraShortestPathsSolver(self.source, 
                                                           self.mfd_instance.adj_mat, 
                                                           weight_mat, 
                                                           mode = mode,
                                                           matrix_representation = self.matrix_representation)
        # Run dijkstra
        self.dijkstra_solver.run_dijkstra()
        # Construct and return the subgraph of shortest path (which may or may not be a DAg according to the value of self.path_selector_type)
        self.dijkstra_solver.construct_DAG_shortest_path (self.destination)
        return self.dijkstra_solver.dagsp




##########
#####################################  Subgraph Constructor for Transition function  #####################################
##########
class SubGraphConstructorTransF(SubGraphConstructor):

    def __init__(self, path_selector_type, mfd_instance, max_trans_func_successor = False, matrix_representation = True):
        self.path_selector_type = path_selector_type
        self.mfd_instance = mfd_instance
        self.max_trans_func_successor = max_trans_func_successor
        self.matrix_representation = matrix_representation


    def check_filtering_cond_successors (self, cur_elem, next_node, first_check = False):
        return True
    

    def check_filtering_cond_predecessors (self, cur_elem, prec_node):
        return True
    

    def _exist_path (self):
        return any(has_arc(self.subg_s_d, self.source, node) for node in range(len(self.subg_s_d)))
        
    
    def construct_tree_arcbfs (self,
                            source, 
                            graph, 
                            trans_func, 
                            trans_from_sources, 
                            max_trans_func_successor):
        """
            !!! max_trans_func_successor must be set to False if we use RLbased desaggregation algorithms !!!
        """
        # Initalizations, create and push to the queue the outgoing arcs from 'source' 
        queue = []
        subgraph = create_isolated_nodes_graph(len(graph), matrix_representation = self.matrix_representation)
        visited = init_graph_arc_attribute_vals(graph, init_val = False)
        max_val_flow_arc = -1 if not max_trans_func_successor else max(trans_from_sources[source][succ_arc] for succ_arc in trans_from_sources[source])
        for succ_arc in trans_from_sources[source]:
            if has_arc(graph, succ_arc[0], succ_arc[1]) and\
                trans_from_sources[source][succ_arc] > 0 and\
                    trans_from_sources[source][succ_arc] >= max_val_flow_arc and\
                        self.check_filtering_cond_successors (source, 
                                                                succ_arc[1], 
                                                                first_check = True): 
                queue.append((succ_arc[0], succ_arc[1]))
                add_arc(subgraph, succ_arc[0], succ_arc[1])
                visited[succ_arc[0]][succ_arc[1]] = True

        # Loop as long as queue is not empty
        while queue:
            # Pop an arc from queue
            arc = queue.pop(0)

            # Check successors of 'arc' and push them if unvisited
            max_val_flow_arc = -1 if not max_trans_func_successor else max(trans_func[arc][succ_arc] for succ_arc in trans_func[arc])
            for succ_arc in trans_func[arc]:
                if has_arc(graph, succ_arc[0], succ_arc[1]) and\
                    trans_func[arc][succ_arc] > 0 and\
                        trans_func[arc][succ_arc] >=  max_val_flow_arc and\
                            not visited[succ_arc[0]][succ_arc[1]] and\
                                self.check_filtering_cond_successors (arc, 
                                                                        succ_arc[1]):
                    queue.append((succ_arc[0], succ_arc[1]))
                    add_arc(subgraph, succ_arc[0], succ_arc[1])
                    visited[succ_arc[0]][succ_arc[1]] = True

        return subgraph


    def construct_antitree_arcbfs (self,
                                destination, 
                                graph, 
                                trans_func, 
                                trans_to_destinations):
        # Initalizations, create and push to the queue the outgoing arcs from 'source' 
        queue = []
        subgraph = create_isolated_nodes_graph(len(graph), matrix_representation = self.matrix_representation)
        visited = init_graph_arc_attribute_vals(graph, init_val = False)
        for pred_arc in trans_to_destinations[destination]:
            if has_arc(graph, pred_arc[0], pred_arc[1]) and\
                    trans_to_destinations[destination][pred_arc] > 0: 
                queue.append((pred_arc[0], pred_arc[1]))
                add_arc(subgraph, pred_arc[0], pred_arc[1])
                visited[pred_arc[0]][pred_arc[1]] = True

        # Loop as long as the queue is not empty
        while queue:
            # Pop an arc from queue
            arc = queue.pop(0)

            # Construct predecessors of arc
            predecessors = [pred_arc for pred_arc in trans_func if arc in trans_func[pred_arc] and\
                                                                trans_func[pred_arc][arc] > 0 and\
                                                                has_arc(graph, pred_arc[0], pred_arc[1]) and\
                                                                not visited[pred_arc[0]][pred_arc[1]] and\
                                                                self.check_filtering_cond_predecessors (arc, 
                                                                                                        pred_arc[0])]

            # Check successors of 'arc' and push them if unvisited
            for pred_arc in predecessors:
                queue.append((pred_arc[0], pred_arc[1]))
                add_arc(subgraph, pred_arc[0], pred_arc[1])
                visited[pred_arc[0]][pred_arc[1]] = True

        return subgraph


    def subgraph_source_dest (self, source, destination):
        self.source, self.destination = source, destination
        self.subg_s_d = self.construct_antitree_arcbfs (self.destination, 
                                                        self.construct_tree_arcbfs (self.source, 
                                                                                self.mfd_instance.adj_mat, 
                                                                                self.mfd_instance.transition_function, 
                                                                                self.mfd_instance.transition_from_sources,
                                                                                self.max_trans_func_successor), 
                                                        self.mfd_instance.transition_function, 
                                                        self.mfd_instance.transition_to_destinations)
        return self.subg_s_d
    


class SubGraphConstructorRL (SubGraphConstructorTransF):
    def __init__ (self, path_selector_type, mfd_instance, mfd_solver, matrix_representation = True):
        super().__init__(path_selector_type, mfd_instance, 
                         max_trans_func_successor = False,
                         matrix_representation = matrix_representation)
        # The class does not support UCB and Hierarchical pursuit algorithms
        if mfd_solver.path_selector.dict_parameters["ag_type"] == "UCB" or\
            mfd_solver.path_selector.dict_parameters["ag_type"] == "Hier_cont_pursuit":
            print("This agent type is not yet supported by the class 'SubGraphFiltererRL'")
            sys.exit()
        
        self.mfd_solver = mfd_solver
        self.ag_type = mfd_solver.path_selector.dict_parameters["ag_type"]


    def check_filtering_cond_successors (self, cur_elem, next_node, first_check = False):
        if first_check:
            agent = self.mfd_solver.path_selector.source_agents[(self.source, self.destination)]
        else:
            path_type_name = self.mfd_solver.path_selector.path_selector_type
            if path_type_name == "rl_arc_based":
                agent = self.mfd_solver.path_selector.agents[(self.source, self.destination)+cur_elem]
            else: 
                agent = self.mfd_solver.path_selector.agents[(self.source, self.destination, cur_elem[1])]
        
        if self.ag_type in POLICY_BASED_TYPE_AGENTS or self.ag_type in MIXED_TYPE_AGENTS:
            return agent.policy[agent.actions_to_inds[next_node]] > 0    
        elif self.ag_type in VALUE_BASED_TYPE_AGENTS:
            return True
    

    def check_filtering_cond_predecessors (self, cur_elem, prec_node):
        path_type_name = self.mfd_solver.path_selector.path_selector_type
        if path_type_name == "rl_arc_based":
            agent = self.mfd_solver.path_selector.agents[(self.source, self.destination, prec_node, cur_elem[0])]    
        else:
            agent = self.mfd_solver.path_selector.agents[(self.source, self.destination, prec_node)]

        if self.ag_type in POLICY_BASED_TYPE_AGENTS or self.ag_type in MIXED_TYPE_AGENTS:
            if path_type_name == "rl_arc_based":
                return agent.policy[agent.actions_to_inds[cur_elem[1]]] > 0
            elif path_type_name == "rl_node_based":
                return agent.policy[agent.actions_to_inds[cur_elem[0]]] > 0
        elif self.ag_type in VALUE_BASED_TYPE_AGENTS:
            return True