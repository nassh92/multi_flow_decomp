import random

from copy import deepcopy

import pprint

import sys

import os
sys.path.append(os.getcwd())
from utils.display_utils import print_log

from utils.graph_utils import make_path_simple, path_intersection, check_arc_inclusion

from utils.random_utils import weighted_shuffle

from trans_func_utils import subgraph_source_dest

from utils.shortest_path_solvers import PATH_TYPES, MODES, DijkstraShortestPathsSolver



###########################################################################################################################################################  
### A Multi Flow Desaggregation Class where a (single) super source and a (single) super destination are construted to solve the problem
### For each source, an arc is added to link the source to the super source. Same with the super desination, an arc is added
### to link each destination to the superdestination.
### The desaggregation algorithm does ont chose explicitly the pairs, the pairs are chosen indirectly by choosing the path from
### the super source through the super destination.
###########################################################################################################################################################

class MultiFlowDesagSolverSSSD ():
    # !!! We consider that ALL the matrices have the 'supersource' and 'superdestination' nodes. !!!
    # This concerns : self.adj_mat,... and also 'multi_flow' created later  

    def __init__(self, adj_mat, aggregated_flow, transport_times, pairs, flow_values, 
                 update_transport_time = False, maximal_flow_amount = float("inf")):
        # The adjacency matrix
        self.adj_mat = deepcopy(adj_mat)

        # The aggregated flow and unattributed flow
        self.original_aggregated_flow = aggregated_flow
        self.aggregated_flow = deepcopy(aggregated_flow)
        self.unattributed_flow = [[0 for v in range(len(aggregated_flow))] for u in range(len(aggregated_flow))]

        # Create time related informations
        self.ideal_transport_times = deepcopy(transport_times)
        self.transport_times = deepcopy(transport_times)
        self.update_transport_time = update_transport_time

        # The source-destination pairs
        self.pairs = deepcopy(pairs)

        # The flow values associated to the pairs
        self.original_flow_values = deepcopy(flow_values)
        self.remaining_flow_values = deepcopy(flow_values)
        self.generated_flow_values = [0]*len(flow_values)

        # Process the supersource
        self.super_source = len(adj_mat)-2

        # Process the superdestination
        self.super_destination = len(adj_mat)-1

        # Maximal flow amount to be subtracted from the graph on each iteration
        self.maximal_flow_amount = maximal_flow_amount


    def _correct_weight_matrix (self, weight_mat, mode):
        if mode not in MODES:
            print("Mode is unrecognized.")
            sys.exit()
        if mode == "min_distance":
            #weight_val = 1000000
            # Minimal transport time
            minimal_cost = min([min(row[:-2]) for row in self.transport_times[:-2]])

            # Process 'source_outdegree' and 'destination_indegree'
            source_outdegree = len([self.adj_mat[self.super_source][v] for v in range(len(self.adj_mat))])
            destination_indegree = len([self.adj_mat[v][self.super_destination] for v in range(len(self.adj_mat))])

            # Process the total mass
            source_mass = source_outdegree * minimal_cost
            desintation_mass = destination_indegree * minimal_cost

            # Process the total flow
            total_flow = sum(self.original_flow_values)

            # Correct the matrix with the new val
            for i in range(len(weight_mat)):
                weight_mat[-2][i] = (self.original_aggregated_flow[-2][i] / total_flow) * source_mass\
                                     if 0 < weight_mat[-2][i] and weight_mat[-2][i] != float('inf') else weight_mat[-2][i]
                self.ideal_transport_times[-2][i] = weight_mat[-2][i]
                weight_mat[i][-1] = (self.original_aggregated_flow[i][-1] / total_flow) * desintation_mass\
                                    if 0 < weight_mat[i][-1] and weight_mat[i][-1] != float('inf') else weight_mat[i][-1]
                self.ideal_transport_times[i][-1] = weight_mat[i][-1]

    
    def _return_update_time_target (self, node1, node2): 
        return self.ideal_transport_times[node1][node2] * (self.original_aggregated_flow[node1][node2] / self.aggregated_flow[node1][node2]) 


    def _update_flow_infos(self, path, multi_flow, flow_amount, id_pair = None):
        # Update the flow value associated to the chosen pair (if it is given)
        if id_pair is not None:
            corrected_flow_amount = min(self.remaining_flow_values[id_pair], flow_amount)
            self.remaining_flow_values[id_pair] -= corrected_flow_amount
            self.generated_flow_values[id_pair] += corrected_flow_amount

        # Update the flow/adjacency matrix at the start of the path
        self.aggregated_flow[path[0]][path[1]] -= flow_amount
        self.adj_mat[path[0]][path[1]] = int(bool(self.aggregated_flow[path[0]][path[1]] != 0))
        if self.adj_mat[path[0]][path[1]] and self.update_transport_time: 
            self.transport_times[path[0]][path[1]] = self._return_update_time_target(path[0], path[1])

        # Update the flow/adjacency matrix at the end of the path
        self.aggregated_flow[path[-2]][path[-1]] -= flow_amount
        self.adj_mat[path[-2]][path[-1]] = int(bool(self.aggregated_flow[path[-2]][path[-1]] != 0))
        if self.adj_mat[path[-2]][path[-1]] and self.update_transport_time: 
            self.transport_times[path[-2]][path[-1]] = self._return_update_time_target(path[-2], path[-1])

        #  Update the flow/adjacency matrix at the rest of the path
        for i in range(1, len(path)-2):
            # Decrease the flow on the current arc of the aggregated flow 
            self.aggregated_flow[path[i]][path[i+1]] -= flow_amount

            # Augment the flow on the current arc of of the desagregated constructed flow (if the associated pair is known)
            # else augment the flow on the path in the' non_attributed flow' datastructure 
            if id_pair is not None: 
                multi_flow[id_pair][path[i]][path[i+1]] += flow_amount
            else: 
                self.unattributed_flow[path[i]][path[i+1]] += flow_amount
            
            # Set the current arc to 0 if the aggregated flow on this arc is 0 
            self.adj_mat[path[i]][path[i+1]] = int(bool(self.aggregated_flow[path[i]][path[i+1]]) != 0)

            # Update the time on the current arc if it is still there
            if self.adj_mat[path[i]][path[i+1]] and self.update_transport_time: 
                self.transport_times[path[i]][path[i+1]] = self._return_update_time_target(path[i], path[i+1])


    def _update_weight_matrix (self, weight_mat, path, mode):
        # Sanity check
        if mode not in MODES:
            print("Mode is unrecognized.")
            sys.exit()
        # Update the weight matrix according to the vakue of 'mode'
        if mode == "min_distance":
            #  Update the weight matrix at the rest of the path
            for i in range(0, len(path)-1):
                weight_mat[path[i]][path[i+1]] = float("inf") if self.adj_mat[path[i]][path[i+1]] == 0 else self.transport_times[path[i]][path[i+1]]

        elif mode == "max_capacity":
            #  Update the weight matrix at the rest of the path
            for i in range(0, len(path)-1):
                weight_mat[path[i]][path[i+1]] = 0 if self.adj_mat[path[i]][path[i+1]] == 0 else self.aggregated_flow[path[i]][path[i+1]]
    

    def _exist_path (self, dijkstra_solver):
        if dijkstra_solver.mode == "min_distance":
            return dijkstra_solver.path_estimates[self.super_destination] != float("inf")
        
        elif dijkstra_solver.mode == "max_capacity":
            return dijkstra_solver.path_estimates[self.super_destination] != 0


    def heuristic_multi_flow_desagregation (self, path_type, mode, 
                                            keep_logs = False, seed = 42, 
                                            dir_save_name = None, show = False):
        # If 'keep_logs' is True, initalize the log
        if keep_logs: logs = []

        # Sanity checks
        if mode not in MODES or path_type not in PATH_TYPES:
            print("Mode or path types are unrecognized.")
            sys.exit()

        # Create a list of matrices (which will contain the multiflow to be constructed)
        multi_flow = [[[0 for col in range(len(self.adj_mat))] for row in range(len(self.adj_mat))] for _ in range(len(self.pairs))]

        # Create a djikstra instance (with a corrected weight_matrice)
        weight_mat = deepcopy(self.transport_times) if mode == "min_distance" else deepcopy(self.aggregated_flow) if mode == "max_capacity" else None
        self._correct_weight_matrix (weight_mat, mode)
        dijkstra_solver = DijkstraShortestPathsSolver(self.super_source, 
                                                      self.adj_mat, 
                                                      weight_mat, 
                                                      mode = mode, 
                                                      optional_infos = {"super_destination":self.super_destination,
                                                                        "arc_filtering_criteria":"source_ndestination"})

        # Run the dijkstra algorithm on the solver
        dijkstra_solver.run_dijkstra()
        
        if keep_logs: logs.append([None, None, deepcopy(self.adj_mat), deepcopy(self.aggregated_flow), deepcopy(weight_mat)])
        
        # Main Loop
        it = 0
        while self._exist_path (dijkstra_solver):
            # Construct path
            path = dijkstra_solver.return_path(self.super_destination, path_type = path_type)
            # Take the associated origin/destination pair of the path
            id_pair = self.pairs.index((path[1], path[-2])) if (path[1], path[-2]) in self.pairs else None
            # Process capacity of the path
            flow_capacity = min(self.aggregated_flow[path[i]][path[i+1]] for i in range(len(path)-1))
            # Update the flow on the path (on teh aggregated flow and on the multiflow)
            self._update_flow_infos(path, multi_flow, flow_capacity, id_pair)
            # Update the weight matrix in the Dijkstra algorithm
            self._update_weight_matrix (dijkstra_solver.weight_mat, path, mode)
            # Keep logs if 'keep_logs'
            if keep_logs: logs.append([flow_capacity, path, deepcopy(self.adj_mat), deepcopy(self.aggregated_flow), deepcopy(weight_mat)])
            # re-run the dijkstra algorithm
            dijkstra_solver.run_dijkstra()
            #print("Estimates ", [self.aggregated_flow[i][51] for i in range(len(self.aggregated_flow))])
            it += 1
        #print("Number of iterations is ", it)

        if keep_logs : 
            print_log(logs, seed, dir_save_name, 
                      multi_flow, self.aggregated_flow, self.generated_flow_values, 
                      show = show)

        return multi_flow, self.generated_flow_values




###########################################################################################################################################################  
### A Multi Flow Desaggregation Class where there is (naturally) multiple sources and multiple destinations 
### The desaggregation algorithm choses a pair following a given criteria (for example based on flow value), then proceeds to chose
### the path from the chosen source to the chosen destination.
###########################################################################################################################################################

# The criterias used to chose the pairs
#PAIRS_CRITERIAS = {"max_remaining_flow_val", "proba_remaining_flow_cap"}
PAIRS_CRITERIAS = {"max_remaining_flow_val", "pair_best_path", ("max_remaining_flow_val", "pair_best_path")}

NB_TOTAL_ITERATIONS = 0
NB_EQUAL_FLOW_VAL = 0


def return_nb_total_it():
    return NB_TOTAL_ITERATIONS


def return_nb_equal_flow_val():
    return NB_EQUAL_FLOW_VAL


class MultiFlowDesagSolverMSMD ():
    
    def __init__(self, adj_mat, aggregated_flow, 
                 transport_times, pairs, flow_values, 
                 update_transport_time = False,
                 maximal_flow_amount = float("inf")):
        # Check for zero flow values
        for f_val in flow_values:
            if f_val == 0:
                print("There is a zero flow value.")
                print(flow_values)
                sys.exit()

        # The adjacency matrix
        self.adj_mat = deepcopy(adj_mat)

        # The aggregated flow and unattributed flow
        self.original_aggregated_flow = aggregated_flow
        self.aggregated_flow = deepcopy(aggregated_flow)
        self.unattributed_flow = [[0 for v in range(len(aggregated_flow))] for u in range(len(aggregated_flow))]

        # Create time related informations
        self.ideal_transport_times = deepcopy(transport_times)
        self.transport_times = deepcopy(transport_times)
        self.update_transport_time = update_transport_time

        # The source-destination pairs along with the number of excluded pairs
        self.pairs = deepcopy(pairs)
        self.remaining_pairs = deepcopy(pairs)

        # The flow values associated to the pairs
        self.original_flow_values = deepcopy(flow_values)
        self.remaining_flow_values = deepcopy(flow_values)
        self.generated_flow_values = [0]*len(flow_values)
        
        # Maximal flow amount to be subtracted from the graph on each iteration
        self.maximal_flow_amount = maximal_flow_amount


    def chose_pair(self, pair_criteria):
        global NB_EQUAL_FLOW_VAL
        """
            Select a 'candidate' pair from amongts the remaining pairs using a local criteria
        """
        if pair_criteria == "max_remaining_flow_val":
            max_remaining_flow_val = max(self.remaining_flow_values[i] for i in range(len(self.remaining_flow_values))\
                                          if self.pairs[i] in self.remaining_pairs)
            max_remaining_flow_val_pairs = [self.pairs[i] for i in range(len(self.pairs))\
                                            if self.pairs[i] in self.remaining_pairs and\
                                            self.remaining_flow_values[i] == max_remaining_flow_val]
            chosen_pair = max_remaining_flow_val_pairs[random.randint(0, len(max_remaining_flow_val_pairs)-1)]
            if len(max_remaining_flow_val_pairs) > 1:
                NB_EQUAL_FLOW_VAL += 1
            return chosen_pair
        

    def search_pair(self, weight_mat, mode, pair_criteria):
        """
        Search for a pair for which there is a path linking the source to the destination with the selected criteria.
        Return it if any, return None otherwise
        """
        if pair_criteria not in PAIRS_CRITERIAS:
            print("The pair criteria used is unrecognized.")
            sys.exit()
        
        if pair_criteria not in {"pair_best_path", ("max_remaining_flow_val", "pair_best_path")}: # The cases where only local criterias are used to select pairs
            while len(self.remaining_pairs) > 0:
                # Return a candidate source/destination pair (following the selected criteria)
                source, destination = self.chose_pair(pair_criteria)
                # Construct a 'DijkstraShortestPathsSolver' instance for the source in the selected pair
                dijkstra_solver = DijkstraShortestPathsSolver(source,
                                                              self.adj_mat, 
                                                              weight_mat, 
                                                              mode = mode)
                # Run the dijkstra algorithm on the solver
                dijkstra_solver.run_dijkstra()
                # Return the candidate pair if there is a path linking the source to the destination and remove the pair otherwise 
                if not self._exist_path(dijkstra_solver, destination): 
                    self.remaining_pairs.remove((source, destination))
                else:
                    return (source, destination), dijkstra_solver
            # Return None if no pair is found
            return None
        
        else: 
            # Global criterias are used here. Select the pair where its associated best path is the best among the pairs. 
            # Here "best" depend on the 'mode' chosen.
            # Intializations  
            best_path_estimate, pairs_to_remove, infos_pairs = None, [], []
            # Search for the best remaining pair
            for remain_pair in self.remaining_pairs:
                # Unpack the current pair
                source, destination = remain_pair
                # Create a dijkstra instance associated to the source of the current pair
                dijkstra_solver = DijkstraShortestPathsSolver(source,
                                                              self.adj_mat,
                                                              weight_mat,
                                                              mode = mode)
                # Run the dijkstra algorithm for the source
                dijkstra_solver.run_dijkstra()
                # If there is no path connecting the source to the destination,
                # add the current pair to the list of pairs to be removed
                if not self._exist_path(dijkstra_solver, destination):
                    pairs_to_remove.append(remain_pair)
                
                else: 
                    # else set the selected pair to the current pair if its estimate is better than that of the best one found so far
                    estimate = dijkstra_solver.path_estimates[destination]
                    
                    # then store the pair in 'info_pairs'
                    if pair_criteria == "pair_best_path":
                        if best_path_estimate is None or dijkstra_solver._is_better_than(estimate, best_path_estimate):
                            best_path_estimate = estimate
                        
                        infos_pairs.append((remain_pair, dijkstra_solver, estimate))

                    elif pair_criteria == ("max_remaining_flow_val", "pair_best_path"):
                        id_remain_pair = [i for i in range(len(self.pairs)) if self.pairs[i] == remain_pair][0]
                        infos_pairs.append((remain_pair, dijkstra_solver, estimate, self.remaining_flow_values[id_remain_pair]))
            
            # Return None if there is no pair where endpoints are connected
            if len(pairs_to_remove) == len(self.remaining_pairs):
                return None
            
            # Remove all the pairs in 'pairs_to_remove' from 'self.remaining_pairs'
            for pair in pairs_to_remove:
                self.remaining_pairs.remove(pair)
            
            # Process the selected pair according to the given criteria and return it
            if pair_criteria == "pair_best_path":
                list_candidate_pairs = [elem for elem in infos_pairs if elem[2] == best_path_estimate]
                selected_pair = list_candidate_pairs[random.randint(0, len(list_candidate_pairs)-1)]
            
            elif pair_criteria == ("max_remaining_flow_val", "pair_best_path"):
                max_remaining_flow_val = max(self.remaining_flow_values[i] for i in range(len(self.pairs))
                                                                                if self.pairs[i] in self.remaining_pairs)
                max_remaining_flow_val_pairs = [self.pairs[i] for i in range(len(self.pairs))
                                                                if self.pairs[i] in self.remaining_pairs and\
                                                                    self.remaining_flow_values[i] == max_remaining_flow_val]
                best_path_estimate = max(elem[2] for elem in infos_pairs if elem[0] in max_remaining_flow_val_pairs)
                list_candidate_pairs = [elem for elem in infos_pairs if elem[2] == best_path_estimate]
                selected_pair = list_candidate_pairs[random.randint(0, len(list_candidate_pairs)-1)]

            return selected_pair[0], selected_pair[1]
    
        
    def _return_update_time_target (self, node1, node2): 
        return self.ideal_transport_times[node1][node2] * (self.original_aggregated_flow[node1][node2] / self.aggregated_flow[node1][node2]) 


    def _update_flow_infos(self, path, multi_flow, flow_amount, id_pair):
        # Update the flow value associated to the chosen pair
        corrected_flow_amount = min(self.remaining_flow_values[id_pair], flow_amount)
        self.remaining_flow_values[id_pair] -= corrected_flow_amount
        self.generated_flow_values[id_pair] += corrected_flow_amount

        #  Update the flow/adjacency matrix at the rest of the path
        for i in range(0, len(path)-1):
            # Decrease the flow on the current arc of the aggregated flow 
            self.aggregated_flow[path[i]][path[i+1]] -= flow_amount
            # Augment the flow on the current arc of of the desagregated constructed flow (if the associated pair is known)
            multi_flow[id_pair][path[i]][path[i+1]] += flow_amount
            # Set the current arc to 0 if the aggregated flow on this arc is 0 
            self.adj_mat[path[i]][path[i+1]] = int(bool(self.aggregated_flow[path[i]][path[i+1]]) != 0)
            # Update the time on the current arc if it is still there
            if self.adj_mat[path[i]][path[i+1]] and self.update_transport_time: 
                self.transport_times[path[i]][path[i+1]] = self._return_update_time_target(path[i], path[i+1])


    def _update_weight_matrix (self, weight_mat, path, mode):
        # Sanity check
        if mode not in MODES:
            print("Mode is unrecognized.")
            sys.exit()
        # Update the weight matrix according to the vakue of 'mode'
        if mode == "min_distance":
            #  Update the weight matrix at the rest of the path
            for i in range(0, len(path)-1):
                weight_mat[path[i]][path[i+1]] = float("inf") if self.adj_mat[path[i]][path[i+1]] == 0 else self.transport_times[path[i]][path[i+1]]

        elif mode == "max_capacity":
            #  Update the weight matrix at the rest of the path
            for i in range(0, len(path)-1):
                weight_mat[path[i]][path[i+1]] = 0 if self.adj_mat[path[i]][path[i+1]] == 0 else self.aggregated_flow[path[i]][path[i+1]]
    

    def _exist_path (self, dijkstra_solver, destination):
        if dijkstra_solver.mode == "min_distance":
            return dijkstra_solver.path_estimates[destination] != float("inf")
        
        elif dijkstra_solver.mode == "max_capacity":
            return dijkstra_solver.path_estimates[destination] != 0


    def heuristic_multi_flow_desagregation (self, path_type, mode, 
                                            pair_criteria, keep_logs = False, 
                                            seed = 42, dir_save_name = None, show = False):
        global NB_TOTAL_ITERATIONS
        # If 'keep_logs' is True, initalize the log
        if keep_logs: logs = []

        # Sanity checks
        if mode not in MODES or path_type not in PATH_TYPES:
            print("Mode or path types are unrecognized.")
            sys.exit()

        # Create a list of matrices (which will contain the multiflow to be constructed)
        multi_flow = [[[0 for col in range(len(self.adj_mat))] for row in range(len(self.adj_mat))] for _ in range(len(self.pairs))]

        # Create a djikstra instance (with a corrected weight_matrice)
        weight_mat = deepcopy(self.transport_times) if mode == "min_distance" else deepcopy(self.aggregated_flow) if mode == "max_capacity" else None
        remain_pair = self.search_pair(weight_mat, mode, pair_criteria)

        if keep_logs: logs.append([None, None, deepcopy(self.adj_mat), deepcopy(self.aggregated_flow), deepcopy(weight_mat)])
        
        # Main Loop
        it = 0
        while remain_pair is not None:
            # Unpack the results returned by the last call of 'search_pair'
            (source, destination), dijkstra_solver = remain_pair
            # Construct path
            path = dijkstra_solver.return_path(destination, 
                                               path_type = path_type)
            
            # Process capacity of the path
            flow_capacity = min(min(self.aggregated_flow[path[i]][path[i+1]] for i in range(len(path)-1)),
                                self.maximal_flow_amount)
            # Update the flow on the path (on teh aggregated flow and on the multiflow)
            id_pair = [i for i in range(len(self.pairs)) if (source, destination) == (self.pairs[i][0], self.pairs[i][1])][0]
            self._update_flow_infos(path, 
                                    multi_flow, 
                                    flow_capacity, 
                                    id_pair = id_pair)
            # Update the weight matrix in the Dijkstra algorithm
            self._update_weight_matrix (dijkstra_solver.weight_mat, path, mode)

            # Keep logs if 'keep_logs'
            if keep_logs: logs.append([flow_capacity, path, deepcopy(self.adj_mat), deepcopy(self.aggregated_flow), deepcopy(weight_mat)])

            # Update the remaining pairs and search for a new valid pair
            if self.remaining_flow_values[id_pair] == 0: self.remaining_pairs.remove((source, destination)) 
            remain_pair = self.search_pair(dijkstra_solver.weight_mat, mode, pair_criteria)

            #print("Estimates ", [self.aggregated_flow[i][51] for i in range(len(self.aggregated_flow))])
            it += 1
            NB_TOTAL_ITERATIONS += 1
        #print("Number of iterations is ", it)

        if keep_logs : 
            print_log(logs, seed, dir_save_name, 
                      multi_flow, self.aggregated_flow, self.generated_flow_values, 
                      show = show)

        return multi_flow, self.generated_flow_values



###########################################################################################################################################################  
### A subclass of the class 'MultiFlowDesagSolverMSMD' :
### The main difference with 'MultiFlowDesagSolverMSMD' is that in sublass 'MultiFlowDesagSolverMSMDTransFunc', the paths are selected
### using a transition functions which are given as an attribute of the class.
### The desaggregation algorithm choses a pair following a given criteria (for example based on flow value), then proceeds to chose
### the path from the chosen source to the chosen destination like mentioned previously.
###########################################################################################################################################################

class MultiFlowDesagSolverMSMDTransFunc(MultiFlowDesagSolverMSMD):

    def __init__(self, adj_mat, aggregated_flow, transport_times, pairs, flow_values, 
                 ls_transition_function,
                 update_transport_time = False,
                 update_transition_functions = False,
                 paths_use_only_trans_func = False,
                 max_trans_func_successor = False,
                 nb_max_tries = 1,
                 max_path_length = 1000, 
                 maximal_flow_amount = float("inf")):
        # Create an instance of the parent class
        super().__init__(adj_mat, aggregated_flow, 
                         transport_times, pairs, flow_values, 
                         update_transport_time = update_transport_time, 
                         maximal_flow_amount = maximal_flow_amount)
        
        # Set the transition function
        self.transition_function = deepcopy(ls_transition_function[0])
        self.transition_from_sources = deepcopy(ls_transition_function[1])
        self.transition_to_destinations = deepcopy(ls_transition_function[2])
        self.update_transition_functions = update_transition_functions
        self.nb_max_tries = nb_max_tries
        self.max_path_length = max_path_length

        # Set a boolean 'paths_use_only_trans_func' which is true iif only the transition matrix is used in the path selection algorithm 
        self.paths_use_only_trans_func = paths_use_only_trans_func
        # Set a boolean 'max_trans_func_successor' which is true iif we want the next arc to be selected in a path
        # to correspond to the one which has the highest value on the transition matrix  
        # if 'paths_use_only_trans_func' is False, 'max_trans_func_successor' is ignored
        self.max_trans_func_successor = max_trans_func_successor        


    def _update_flow_infos(self, path, multi_flow, flow_amount, id_pair):
        # Update the flow value associated to the chosen pair
        corrected_flow_amount = min(self.remaining_flow_values[id_pair], flow_amount)
        self.remaining_flow_values[id_pair] -= corrected_flow_amount
        self.generated_flow_values[id_pair] += corrected_flow_amount

        #  Update the flow/adjacency matrix at the rest of the path
        for i in range(0, len(path)-1):
            # Decrease the flow on the current arc of the aggregated flow 
            self.aggregated_flow[path[i]][path[i+1]] -= flow_amount
            # Augment the flow on the current arc of of the desagregated constructed flow (if the associated pair is known)
            multi_flow[id_pair][path[i]][path[i+1]] += flow_amount
            # Set the current arc to 0 if the aggregated flow on this arc is 0 
            self.adj_mat[path[i]][path[i+1]] = int(bool(self.aggregated_flow[path[i]][path[i+1]]) != 0)
            # Update the time on the current arc if it is still there
            if self.adj_mat[path[i]][path[i+1]] and self.update_transport_time: 
                self.transport_times[path[i]][path[i+1]] = self._return_update_time_target(path[i], path[i+1])
            # Update the flow on the corresponding transition function if 'update_transition_functions' is enabled
            if self.update_transition_functions:
                if i == 0:
                    self.transition_from_sources[path[0]][(path[0], path[1])] -= flow_amount
                else:
                    self.transition_function[(path[i-1], path[i])][(path[i], path[i+1])] -= flow_amount
            
            # Sanity check
            if self.aggregated_flow[path[i]][path[i+1]] < 0:
                print("Capacity is negative.", type(self).__name__)
                sys.exit() 

        # Update the flow on the transition list associated to the destination if 'update_transition_functions' is enabled
        if self.update_transition_functions: self.transition_to_destinations[path[-1]][(path[-2], path[-1])] -= flow_amount

    
    ###########################################################   Best effort functions ###########################################
    def return_path_best_effort(self, dijkstra_solver, destination):
        # Construct DAG of shortest paths and use it to construct the nodes wich are part of a path from the source to the destination
        dijkstra_solver.construct_DAG_shortest_path(destination)
        nodes_dagsp = {u for u in range(len(dijkstra_solver.dagsp))
                            for v in range(len(dijkstra_solver.dagsp))
                                if dijkstra_solver.dagsp[u][v] == 1 or dijkstra_solver.dagsp[v][u] == 1}
        
        # Initialization, number of tries, and path length
        cpt_tries = 0
        while cpt_tries < self.nb_max_tries:
            # Intializations of path construction loop
            retry = False
            path_length = 0
            successors_arcs = sorted([succ_arc for succ_arc in self.transition_from_sources[dijkstra_solver.source]
                                                    if self.adj_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                                                        self.transition_from_sources[dijkstra_solver.source][succ_arc] > 0 and\
                                                        succ_arc[0] in nodes_dagsp and succ_arc[1] in nodes_dagsp])
            
            if len(successors_arcs) > 0:
                arc = random.choices(successors_arcs,
                                    weights = [self.transition_from_sources[dijkstra_solver.source][succ_arc] for succ_arc in successors_arcs],
                                    k = 1)[0]
                path = [arc[0], arc[1]]
            else:
                retry = True
                cpt_tries += 1

            # Main loop, chose a successor node stochasticaly according to the 'transition_function and add it to 'path'
            while not retry and arc[1] != destination:
                successors_arcs = sorted([succ_arc for succ_arc in self.transition_function[arc]
                                                        if self.adj_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                                                            self.transition_function[arc][succ_arc] > 0 and\
                                                            succ_arc[0] in nodes_dagsp and succ_arc[1] in nodes_dagsp])
                if len(successors_arcs) > 0 and path_length < self.max_path_length:
                    arc = random.choices(successors_arcs,
                                        weights = [self.transition_function[arc][succ_arc] for succ_arc in successors_arcs],
                                        k = 1)[0]
                    path.append(arc[1])
                    path_length += 1
                else:
                    retry = True
                    cpt_tries += 1
            
            if not retry: return make_path_simple (path)
        
        return dijkstra_solver.return_path(destination, path_type = "random")
    

    ###########################################################   Functions using transition function #######################################
    def _chose_successor_arc_trans_func(self, source, arc, graph_mat):
        """
        Return successor arc using the transition function 
        """
        if arc is None:
            successor_arcs = [succ_arc for succ_arc in self.transition_from_sources[source] if graph_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                                                                                        self.transition_from_sources[source][succ_arc] > 0]
            chosen_arc = random.choices(
                                successor_arcs,
                                weights = [self.transition_from_sources[source][succ_arc] for succ_arc in successor_arcs],
                                k = 1)[0]
        else:
            successor_arcs = [succ_arc for succ_arc in self.transition_function[arc] if graph_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                                                                                                self.transition_function[arc][succ_arc] > 0]
            chosen_arc = random.choices(
                                successor_arcs,
                                weights = [self.transition_function[arc][succ_arc] for succ_arc in successor_arcs],
                                k = 1)[0]
        """
        if arc is None:
            if not self.max_trans_func_successor:
                successor_arcs = [succ_arc for succ_arc in self.transition_from_sources[source] if graph_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                                                                                            self.transition_from_sources[source][succ_arc] > 0]
                chosen_arc = random.choices(
                                    successor_arcs,
                                    weights = [self.transition_from_sources[source][succ_arc] for succ_arc in successor_arcs],
                                    k = 1)[0]
            else:
                max_succ_flow = max(self.transition_from_sources[source][succ_arc] for succ_arc in self.transition_from_sources[source]
                                                                                        if graph_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                                                                                            self.transition_from_sources[source][succ_arc] > 0)
                successor_arcs = [succ_arc for succ_arc in self.transition_from_sources[source] if graph_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                                                                                            self.transition_from_sources[source][succ_arc] > 0 and\
                                                                                                self.transition_from_sources[source][succ_arc] == max_succ_flow]
                chosen_arc = successor_arcs[random.randint(0, len(successor_arcs)-1)]
        
        else:
            if not self.max_trans_func_successor:
                successor_arcs = [succ_arc for succ_arc in self.transition_function[arc] if graph_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                                                                                                self.transition_function[arc][succ_arc] > 0]
                chosen_arc = random.choices(
                                    successor_arcs,
                                    weights = [self.transition_function[arc][succ_arc] for succ_arc in successor_arcs],
                                    k = 1)[0]
            else:
                max_succ_flow = max(self.transition_function[arc][succ_arc] for succ_arc in self.transition_function[arc] 
                                                                                if graph_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                                                                                    self.transition_function[arc][succ_arc] > 0)
                successor_arcs = [succ_arc for succ_arc in self.transition_function[arc] if graph_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                                                                                            self.transition_function[arc][succ_arc] > 0 and\
                                                                                                self.transition_function[arc][succ_arc] == max_succ_flow]
                chosen_arc = successor_arcs[random.randint(0, len(successor_arcs)-1)]
        """
        return chosen_arc
    

    def return_path_trans_func (self, graph_mat, source, destination):
        """
        Return a random path of 'graph_mat' using the transition function
        """
        arc = self._chose_successor_arc_trans_func(source, None, graph_mat)
        path = [arc[0], arc[1]]
        path_length = 0
        while arc[1] != destination and path_length < self.max_path_length:
            arc = self._chose_successor_arc_trans_func(source, arc, graph_mat)
            path.append(arc[1])
            path_length += 1
        
        if path_length == self.max_path_length:
            print("Max path length is attained ", path_length)
            sys.exit()
        
        return make_path_simple (path)


    def search_pair_trans_func(self, weight_mat, mode, pair_criteria):
        """
        Search for a pair for which there is a path linking the source to the destination with the selected criteria.
        Return it if any, return None otherwise
        """
        if pair_criteria not in PAIRS_CRITERIAS:
            print("The pair criteria used is unrecognized.")
            sys.exit()

        if pair_criteria not in {"pair_best_path", ("max_remaining_flow_val", "pair_best_path")}: # The cases where only local criterias are used to select pairs
            while len(self.remaining_pairs) > 0:
                # Return a candidate source/destination pair (following the selected criteria)
                source, destination = self.chose_pair(pair_criteria)
                # Create a subgraphe containig the arcs which lies in a path from 'source' to 'destination'
                subg_s_d = subgraph_source_dest(source, destination, self.adj_mat, 
                                                self.transition_function, 
                                                self.transition_from_sources, 
                                                self.transition_to_destinations,
                                                self.max_trans_func_successor)
                # Return the candidate pair if there is a path linking the source to the destination and remove the pair otherwise 
                if all(subg_s_d[source][node] == 0 for node in range(len(subg_s_d))): 
                    self.remaining_pairs.remove((source, destination))
                else:
                    return (source, destination), subg_s_d
            # Return None if no pair is found
            return None

    
    ##################################################################################################################################
    def return_path (self, source, destination, opt_param):
        # dijkstra_solver, destination
        if not self.paths_use_only_trans_func:
            dijkstra_solver = opt_param["dijkstra_solver"]
            return self.return_path_best_effort(dijkstra_solver, destination)

        else:
            graph_mat = opt_param["graph_mat"]
            return self.return_path_trans_func(graph_mat, source, destination)


    def search_valid_pair(self, weight_mat, mode, pair_criteria):
        """
        Search for a pair for which there is a path linking the source to the destination with the selected criteria.
        Return it if any, return None otherwise
        """
        if not self.paths_use_only_trans_func:
            return self.search_pair(weight_mat, mode, pair_criteria)
        
        else:
            return self.search_pair_trans_func(weight_mat, mode, pair_criteria)
    
    
    def heuristic_multi_flow_desagregation (self, pair_criteria, 
                                            continue_best_effort = False,
                                            init_multi_flow = None, 
                                            keep_logs = False, seed = 42, 
                                            dir_save_name = None, show = False):
        global NB_TOTAL_ITERATIONS
        # If 'keep_logs' is True, initalize the log
        if keep_logs: logs = []

        # Create a list of matrices (which will contain the multiflow to be constructed)
        if init_multi_flow is None:
            multi_flow = [[[0 for col in range(len(self.adj_mat))] for row in range(len(self.adj_mat))] for _ in range(len(self.pairs))]
        else:
            multi_flow = init_multi_flow

        # Create a djikstra instance (with a corrected weight_matrice)
        remain_pair = self.search_valid_pair(self.transport_times, "min_distance", pair_criteria)

        if keep_logs: logs.append([None, None, deepcopy(self.adj_mat), deepcopy(self.aggregated_flow)])
        
        # Main Loop
        it = 0
        while remain_pair is not None:
            # Unpack the results returned by the last call of 'search_pair'
            (source, destination), path_chooser = remain_pair

            # Construct path
            path = self.return_path(source, destination, 
                                    opt_param = {"graph_mat":path_chooser} if self.paths_use_only_trans_func else {"dijkstra_solver":path_chooser})
            
            # Process capacity of the path
            flow_capacity = min(min(self.aggregated_flow[path[i]][path[i+1]] for i in range(len(path)-1)),
                                self.maximal_flow_amount)
            
            # Update the flow on the path (on the aggregated flow and on the multiflow)
            id_pair = [i for i in range(len(self.pairs)) if (source, destination) == (self.pairs[i][0], self.pairs[i][1])][0]
            self._update_flow_infos(path, 
                                    multi_flow, 
                                    flow_capacity, 
                                    id_pair = id_pair)

            # Keep logs if 'keep_logs'
            if keep_logs: logs.append([flow_capacity, path, deepcopy(self.adj_mat), deepcopy(self.aggregated_flow)])

            # Update the remaining pairs and search for a new valid pair
            if self.remaining_flow_values[id_pair] == 0: self.remaining_pairs.remove((source, destination)) 
            remain_pair = self.search_valid_pair(self.transport_times, "min_distance", pair_criteria)

            #print("Estimates ", [self.aggregated_flow[i][51] for i in range(len(self.aggregated_flow))])
            it += 1
            NB_TOTAL_ITERATIONS += 1
        #print("Number of iterations is ", it)

        if keep_logs : 
            print_log(logs, seed, dir_save_name, 
                      multi_flow, self.aggregated_flow, self.generated_flow_values, 
                      show = show)

        # If 'continue_best_effort' is True, apply heuristic with best effort 
        if continue_best_effort:
            self.paths_use_only_trans_func, nb_max_tries = False, self.nb_max_tries
            self.nb_max_tries = 0
            multi_flow, self.generated_flow_values = self.heuristic_multi_flow_desagregation (pair_criteria, 
                                                                                              continue_best_effort = False,
                                                                                              init_multi_flow = multi_flow,
                                                                                              keep_logs = keep_logs, 
                                                                                              seed = seed, 
                                                                                              dir_save_name = dir_save_name, 
                                                                                              show = show)
            self.paths_use_only_trans_func, self.nb_max_tries = True, nb_max_tries
            return multi_flow, self.generated_flow_values 
        
        else:
            return multi_flow, self.generated_flow_values



###########################################################################################################################################################  
### A subclass of the class 'MultiFlowDesagSolverMSMDTransFunc' :
### The main difference with 'MultiFlowDesagSolverMSMDTransFunc' is that in sublass 'MultiFlowDesagSolverMSMDTransFuncMulP', 
### the algorithm selects a path for each valid source-destination pair
### using a transition functions which are given as an attribute of the class.
### The desaggragation algorithm choses a The paths are then ordered randomly and selected one by one. If for a given
### The desaggregation algorithm choses a pair following a given criteria (for example based on flow value), then proceeds to chose
### the path from the chosen source to the chosen destination like mentioned previously.
###########################################################################################################################################################
REORDERING_PAIRS_POLICIES_NAMES = {"uniform", "proba_remaining_flow_val", "remaining_max_flow"}

STOCHASTIC_PAIRS_POLICIES_NAMES = {"uniform", "proba_remaining_flow_val"}

SEMI_STOCHASTIC_PAIRS_POLICIES_NAMES = {"remaining_max_flow"}

class MultiFlowDesagSolverMSMDTransFuncMulP(MultiFlowDesagSolverMSMDTransFunc):

    def __init__(self, adj_mat, aggregated_flow, 
                 transport_times, pairs, flow_values, 
                 ls_transition_function,
                 update_transport_time = False,
                 update_transition_functions = False,
                 paths_use_only_trans_func = False,
                 max_path_length = 1000, 
                 maximal_flow_amount = 1,
                 reodering_pairs_policy_name = "uniform"):
        # Test if weight pair policy exists
        if reodering_pairs_policy_name not in REORDERING_PAIRS_POLICIES_NAMES:
            print("Weight pair policy not recognized.")
            sys.exit()
        
        # Create an instance of the parent class
        super().__init__(adj_mat, aggregated_flow, 
                         transport_times, pairs, flow_values, 
                         ls_transition_function,
                         update_transport_time = update_transport_time,
                         update_transition_functions = update_transition_functions,
                         paths_use_only_trans_func = paths_use_only_trans_func,
                         nb_max_tries = 1,
                         max_path_length = max_path_length, # IMPORTANT for not having an infinite loop in choosing paths ! 
                         maximal_flow_amount = maximal_flow_amount)

        # Set weight pairs policy
        self.reodering_pairs_policy_name = reodering_pairs_policy_name
        if reodering_pairs_policy_name in STOCHASTIC_PAIRS_POLICIES_NAMES:
            self.reodering_pairs_policy = self.WeightPairsPolicy(reodering_pairs_policy_name, self)
        
        elif reodering_pairs_policy_name in SEMI_STOCHASTIC_PAIRS_POLICIES_NAMES:
            self.reodering_pairs_policy = self.SemiStochasticReoderer(reodering_pairs_policy_name, self)


    class WeightPairsPolicy:

        def __init__ (self, reodering_pairs_policy_name, heurs_solver):
             self.reodering_pairs_policy_name = reodering_pairs_policy_name
             self.heurs_solver = heurs_solver
             self.process_pairs_weights()
            
        def process_pairs_weights(self):
            if self.reodering_pairs_policy_name == "uniform":
                self.weights_pairs = [int(fl_val > 0) for fl_val in self.heurs_solver.remaining_flow_values]
            elif self.reodering_pairs_policy_name == "proba_remaining_flow_val":
                self.weights_pairs = [fl_val for fl_val in self.heurs_solver.remaining_flow_values]

        def return_pairs_weights(self):
            return self.weights_pairs


    class SemiStochasticReoderer:
        def __init__ (self, reodering_pairs_policy_name, heurs_solver):
             self.reodering_pairs_policy_name = reodering_pairs_policy_name
             self.heurs_solver = heurs_solver
            
        def chose_ind_candidate(self, ls_infos_rem_flow):
            if self.reodering_pairs_policy_name == "remaining_max_flow":
                rem_max_flow = max(e[1] for e in ls_infos_rem_flow)
                rem_max_flow_ids = [idx for idx, rem_fl_val in ls_infos_rem_flow if rem_fl_val == rem_max_flow]
                return rem_max_flow_ids[random.randint(0, len(rem_max_flow_ids)-1)], rem_max_flow

        def return_reordered_pairs(self, ls_infos_poss_paths):
            ls_infos_rem_flow = [(ind, self.heurs_solver.remaining_flow_values[elem[1]]) for ind, elem in enumerate(ls_infos_poss_paths)]
            reordered_ids_pairs = []
            while len(ls_infos_rem_flow) > 0:
                ind_add, cur_max_flow = self.chose_ind_candidate(ls_infos_rem_flow)
                reordered_ids_pairs.append(ind_add)
                ls_infos_rem_flow.remove((ind_add, cur_max_flow))
            return [ls_infos_poss_paths[ind] for ind in reordered_ids_pairs]


    def _reorder_pairs(self, ls_infos_poss_paths = None):
        if self.reodering_pairs_policy_name in STOCHASTIC_PAIRS_POLICIES_NAMES:
            # !!!!!!!!!!!!!!!!!!!!!!  REODER WITH REPROCESSED WEIGHTS  ??????????????!!!!!!!!!!
            self.reodering_pairs_policy.process_pairs_weights()
            ls_weights = self.reodering_pairs_policy.return_pairs_weights()
            shuffled_indices = weighted_shuffle(list(range(len(ls_infos_poss_paths))), 
                                                [ls_weights[info_pair[1]] for info_pair in ls_infos_poss_paths])
            ls_infos_poss_paths = [ls_infos_poss_paths[indice] for indice in shuffled_indices]
        
        elif self.reodering_pairs_policy_name in SEMI_STOCHASTIC_PAIRS_POLICIES_NAMES:
            ls_infos_poss_paths = self.reodering_pairs_policy.return_reordered_pairs(ls_infos_poss_paths)
        return ls_infos_poss_paths
    

    def _process_conflicts_costs(self, 
                                 id_pair_add, id_pair_test,
                                 i, j,
                                 arcs_intersection, 
                                 ls_infos_poss_paths, 
                                 chosen_pairs,
                                 multi_flow):
        # [(a remaining pair, pair id in 'self.pairs', a path conneting the pair, flow to be added in the path), ...]
        # For each arc test if there is a conflict between 'path_to_add' and 'path_to_test' and update the costs accordingly
        cost_add, cost_test = 0, 0
        for arc in arcs_intersection: 
            # Process the Id pairs (id in self.pairs) for the !remaining! pairs 
            # for which 'arc' belongs to the current 'chosen_paths' ('chosen_id_pairs') or 'path1'
            # ids_valid_pair_arc = [id_pair for id_pair in ids_remaining_pairs if id_pair in chosen_id_pairs+[id_pair_p1] and\
            #                                                                        multi_flow[id_pair][arc[0]][arc[1]] > 0]
            ids_valid_pair_arc = [chosen_pairs[k][1] for k in range(len(chosen_pairs)) if check_arc_inclusion(arc, chosen_pairs[k][2])]+[id_pair_add]
            sum_flow_arc_paths = sum(info_path[3] for info_path in ls_infos_poss_paths if info_path[1] in ids_valid_pair_arc)
            if sum_flow_arc_paths > self.aggregated_flow[arc[0]][arc[1]]:
                if sum_flow_arc_paths - chosen_pairs[j][3] + ls_infos_poss_paths[i][3] <= self.aggregated_flow[arc[0]][arc[1]]:
                    cost_add += multi_flow[id_pair_add][arc[0]][arc[1]]
                    cost_test += multi_flow[id_pair_test][arc[0]][arc[1]]
                else: 
                    return float('inf'), 0
        return cost_add, cost_test
    

    def filter_paths(self, possible_paths, rempairs_flow_amounts, multi_flow):
        # Sanity check for list lengths
        if len(possible_paths) != len(self.remaining_pairs) or len(rempairs_flow_amounts) != len(self.remaining_pairs):
            print(" Length do not correspond to each other.")
            sys.exit()
        
        # Program assumption : 'self.remaining_pairs' has at least one element
        # Initialize a list containing data on the remaining pairs along with their associated weights
        # [(a remaining pair, pair id in 'self.pairs', a path conneting the pair, flow to be added in the path), ...]
        ls_infos_poss_paths = [(self.remaining_pairs[i],
                                self.pairs.index(self.remaining_pairs[i]),
                                possible_paths[i],
                                rempairs_flow_amounts[i]) for i in range(len(self.remaining_pairs))]
        ls_infos_poss_paths = self._reorder_pairs(ls_infos_poss_paths = ls_infos_poss_paths)

        chosen_pairs = [ls_infos_poss_paths[0]]
        for i in range(1, len(ls_infos_poss_paths)):
            path_to_add, id_pair_add, add_path = ls_infos_poss_paths[i][2], ls_infos_poss_paths[i][1], True
            pairs_to_remove = []

            for j in range(len(chosen_pairs)):
                path_to_test, id_pair_test = chosen_pairs[j][2], chosen_pairs[j][1]
                arcs_intersection = path_intersection(path_to_add, path_to_test)
                cost_add, cost_test = self._process_conflicts_costs(id_pair_add, id_pair_test,
                                                                    i, j,
                                                                    arcs_intersection, 
                                                                    ls_infos_poss_paths, 
                                                                    chosen_pairs,
                                                                    multi_flow)
                if cost_add != 0: # if there is a conflict
                    if cost_test <= cost_add: # can't add 'path_to_add', so we don't remove any paths and we go to the next iteration
                        pairs_to_remove = []
                        add_path = False
                        break
                    else: # add path of index j into the path which will be deleted
                        pairs_to_remove.append(deepcopy(chosen_pairs[j]))    
            
            for pair_to_rem in pairs_to_remove:
                chosen_pairs.remove(pair_to_rem)
            
            if add_path: chosen_pairs.append(ls_infos_poss_paths[i])
        
        paths, flow_amounts, id_pairs = [info_pair[2] for info_pair in chosen_pairs], [info_pair[3] for info_pair in chosen_pairs], [info_pair[1] for info_pair in chosen_pairs]
        return paths, flow_amounts, id_pairs
    

    def return_paths(self, multi_flow):
        pairs_to_remove, possible_paths = [], []
        for rema_pair in self.remaining_pairs:
            # Return a candidate source/destination pair (following the selected criteria)
            source, destination = rema_pair
            # Create a subgraphe containig the arcs which lies in a path from 'source' to 'destination'
            subg_s_d = subgraph_source_dest(source, destination, 
                                            self.adj_mat, 
                                            self.transition_function, 
                                            self.transition_from_sources, 
                                            self.transition_to_destinations)
            # Return the candidate pair if there is a path linking the source to the destination and remove the pair otherwise 
            if all(subg_s_d[source][node] == 0 for node in range(len(subg_s_d))): 
                pairs_to_remove.append(rema_pair)
            else:
                possible_paths.append(self.return_path_trans_func(subg_s_d, source, destination))
        
        # Remove every pair from pairs_to_remove
        for pair in pairs_to_remove:
            self.remaining_pairs.remove(pair)

        # Process the flow amouns associated to the remaining pairs
        rempairs_flow_amounts = []
        for i in range(len(self.remaining_pairs)):
            path_pair, id_pair = possible_paths[i], self.pairs.index(self.remaining_pairs[i])
            flow_amount = min(min(self.aggregated_flow[path_pair[j]][path_pair[j+1]] for j in range(len(path_pair)-1)),
                              self.remaining_flow_values[id_pair],
                              self.maximal_flow_amount)
            rempairs_flow_amounts.append(flow_amount)    

        # Return the paths along with their flow amounts and their associated pair id
        return self.filter_paths(possible_paths, rempairs_flow_amounts, multi_flow) if len(possible_paths) > 0 else ([], [], [])
    

    def heuristic_multi_flow_desagregation (self, init_multi_flow = None, 
                                            keep_logs = False, seed = 42, 
                                            dir_save_name = None, show = False):
        global NB_TOTAL_ITERATIONS
        # If 'keep_logs' is True, initalize the log
        if keep_logs: logs = []

        # Create a list of matrices (which will contain the multiflow to be constructed)
        if init_multi_flow is None:
            multi_flow = [[[0 for col in range(len(self.adj_mat))] for row in range(len(self.adj_mat))] for _ in range(len(self.pairs))]
        else:
            multi_flow = init_multi_flow

        if keep_logs: logs.append([deepcopy(self.adj_mat), deepcopy(self.aggregated_flow)])
        
        # Main Loop
        it = 0
        while len(self.remaining_pairs) > 0:
            paths, flow_amounts, id_pairs = self.return_paths(multi_flow)
            
            for path, fl_amount, id_pair in zip(paths, flow_amounts, id_pairs):
                self._update_flow_infos(path, 
                                        multi_flow, 
                                        fl_amount, 
                                        id_pair = id_pair)
                if self.remaining_flow_values[id_pair] == 0: self.remaining_pairs.remove((self.pairs[id_pair][0], self.pairs[id_pair][1]))

            # Keep logs if 'keep_logs'
            if keep_logs: logs.append([deepcopy(self.adj_mat), deepcopy(self.aggregated_flow)])

            #print("Estimates ", [self.aggregated_flow[i][51] for i in range(len(self.aggregated_flow))])
            it += 1
            NB_TOTAL_ITERATIONS += 1
        #print("Number of iterations is ", it)

        if keep_logs : 
            print_log(logs, seed, dir_save_name, 
                      multi_flow, self.aggregated_flow, self.generated_flow_values, 
                      show = show)
        return multi_flow, self.generated_flow_values




######################################################################################################################################
##########################################################   Tests in the main    ####################################################
######################################################################################################################################
if __name__ == "__main__":
    graph_id = 0

    if graph_id == 0: # Décomposition_de_flots_nass_corr, Exemple de la Figure 5
        adj_mat = [[0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [0, 1, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]]
        
        aggregated_flow = [[0, 1, 2, 0, 0, 0],
                           [0, 0, 1, 2, 0, 0],
                           [0, 2, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 3],
                           [3, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]]
        
        transport_times = [[0 if i==j else 1 if adj_mat[i][j]==1 else float("inf") for j in range(len(adj_mat))] for i in range(len(adj_mat))]
        
        mf_decomp = MultiFlowDesagSolverSSSD(adj_mat=adj_mat, 
                                        aggregated_flow=aggregated_flow,
                                        transport_times=transport_times, 
                                        pairs=[(0,3)], 
                                        flow_values=[3],
                                        update_transport_time = False)
        
        multi_flow, flow_vals = mf_decomp.heuristic_multi_flow_desagregation (path_type="first_fit", mode="min_distance",
                                                                              keep_logs=True, dir_name_save="test_heursitic/")
        print("---------------------------------- END Execution ----------------------------------")
    
    elif graph_id == 1: # rapport master, Exemple de la Figure 2
        adj_mat = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        aggregated_flow = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        transport_times = [[0 if i==j else 1 if adj_mat[i][j]==1 else float("inf") for j in range(len(adj_mat))] for i in range(len(adj_mat))]

        mf_decomp = MultiFlowDesagSolverSSSD(adj_mat=adj_mat, 
                                            aggregated_flow=aggregated_flow,
                                            transport_times=transport_times, 
                                            pairs=[(0,7), (0, 8)], 
                                            flow_values=[1, 1],
                                            update_transport_time = True)
        
        multi_flow, flow_vals = mf_decomp.heuristic_multi_flow_desagregation (path_type="random", mode="max_capacity", 
                                                                              keep_logs=True)
        print("---------------------------------- END Execution ----------------------------------")

    elif graph_id == 2: # rapport master, Exemple de la Figure 3, pairs : (1, 4), (2, 1) et (2, 4) (numérotation 1 à 4) ( a vérifier)
        adj_mat = [[0, 0, 0, 1, 0, 1],
                   [1, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]]
        
        aggregated_flow = [[0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 2],
                           [1, 2, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]]
        
        transport_times = [[0 if i==j else 1 if adj_mat[i][j]==1 else float("inf") for j in range(len(adj_mat))] for i in range(len(adj_mat))]

        mf_decomp = MultiFlowDesagSolverSSSD(adj_mat=adj_mat, 
                                            aggregated_flow=aggregated_flow,
                                            transport_times=transport_times, 
                                            pairs=[(0, 3), (1, 0), (1, 3)], 
                                            flow_values=[1, 1, 1],
                                            update_transport_time = True)
        
        multi_flow, flow_vals = mf_decomp.heuristic_multi_flow_desagregation (path_type="random", mode="max_capacity",
                                                                              keep_logs=True)
        print("---------------------------------- END Execution ----------------------------------")

    elif graph_id == 3: # rapport master, Exemple de la Figure 4, (A REFAIRE!!!!!)
        adj_mat = [[0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        aggregated_flow = [[0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        transport_times = [[0 if i==j else 1 if adj_mat[i][j]==1 else float("inf") for j in range(len(adj_mat))] for i in range(len(adj_mat))]

        mf_decomp = MultiFlowDesagSolverSSSD(adj_mat=adj_mat, 
                                            aggregated_flow=aggregated_flow,
                                            transport_times=transport_times, 
                                            pairs=[(0, 4), (5, 2)], 
                                            flow_values=[1, 1],
                                            update_transport_time = False)
        
        multi_flow, flow_vals = mf_decomp.heuristic_multi_flow_desagregation (path_type="random", mode="max_capacity",
                                                                              keep_logs=True)
        print("---------------------------------- END Execution ----------------------------------")
        
        
    print("Les multiflows ")
    pprint.pprint(multi_flow)

    print("Aggregated flow ")
    pprint.pprint(mf_decomp.aggregated_flow)

    print("Les valeurs de flows ")
    pprint.pprint(flow_vals)
    