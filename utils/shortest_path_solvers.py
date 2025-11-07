import sys
import random
from copy import deepcopy
import os
import sys

sys.path.append(os.getcwd())
from graph_utils import make_path_simple, create_isolated_nodes_graph, successors, predecessors, add_arc, has_arc


MODES = ["min_distance", "max_capacity"]


PATH_TYPES = ["first_fit", "random", "random_nfilter_byshortest_path"]
# The path type "random_nfilter_byshortest_path" which is a variant of
# the case 'mode == "max_capacity" and path_type == "random"'

ARC_FILTERING_CRITERIAS = [None, "source_ndestination"]



class DijkstraShortestPathsSolver:

    def __init__(self, source, graph, weights, 
                 mode = "min_distance", 
                 matrix_representation = True, 
                 optional_infos = None):
        # The source, the graph matrix an the weight matrix
        self.source = source
        self.graph = graph
        self.weights = weights
        self.matrix_representation = matrix_representation

        # The mode is how to intrepret the weights as well the estimate of the paths
        if mode not in MODES:
            print("Mode", self.mode, " is not recognized.")
            sys.exit()
        self.mode = mode

        # Optional informations (amon which the 'arc_filtering_criteria')
        self.optional_infos = optional_infos
        self.arc_filtering_criteria = optional_infos.get("arc_filtering_criteria", None) if optional_infos is not None else None
        self.predecessors_list = optional_infos.get("predecessors_list", None) if optional_infos is not None else None

    
    def initialize(self):
        # The processed and unprocessed nodes 
        self.processed = set()
        self.not_processed = set(range(len(self.graph)))
        # Initalize all paths estimates with infinity/0 and predecessors with None
        self.path_estimates = []
        self.predecessors = []
        for _ in range(len(self.graph)):
            self.path_estimates.append(float('inf') if self.mode == "min_distance" else 0 if self.mode == "max_capacity" else None) 
            self.predecessors.append(None)
        
        self.path_estimates[self.source] = 0 if self.mode == "min_distance" else float('inf') if self.mode == "max_capacity" else None


    def _is_better_than(self, estimate1, estimate2):
        if self.mode == "min_distance":
            return estimate1 < estimate2

        elif self.mode == "max_capacity":
            return estimate1 > estimate2
    

    def _process_new_estimate(self, node1, node2):
        if not has_arc(self.graph, node1, node2):
            print("Erreur dans la matrice d'adjacence.")
            sys.exit()

        if self.mode == "min_distance":
            return self.path_estimates[node1] + self.weights[node1][node2]
        
        elif self.mode == "max_capacity":
            return min(self.path_estimates[node1], self.weights[node1][node2])

    
    def _search_min(self):
        # Initialisations
        for e in self.not_processed:
            break
        cur_arg_min = e
        cur_min = self.path_estimates[cur_arg_min]
        # Search the node with minimal estimate of cost value in not_processed
        for node in self.not_processed:
            if self.path_estimates[node] < cur_min:
                cur_min = self.path_estimates[node]
                cur_arg_min = node
        return cur_arg_min
    

    def _search_max(self):
        # Initialisations
        for e in self.not_processed:
            break
        cur_arg_max = e
        cur_max = self.path_estimates[cur_arg_max]
        # Search the node with maximal estimate of cost value in not_processed
        for node in self.not_processed:
            if self.path_estimates[node] > cur_max:
                cur_max = self.path_estimates[node]
                cur_arg_max = node
        return cur_arg_max


    def _filter_arc (self, u, v):
        # Filter the nodes in "self.not_processed" using criteria 'filtering_criteria' 
        # Return an error if filtering criteria is not in 'FILTERING_CRITERIAS'
        if self.arc_filtering_criteria not in ARC_FILTERING_CRITERIAS:
            print("Filtering criteria unrecognized.")
            sys.exit()
        
        if self.arc_filtering_criteria is None:
            return False
        
        else:
            if self.arc_filtering_criteria == "source_ndestination":
                return self.predecessors[u] == self.source and v == self.optional_infos["super_destination"]


    def run_dijkstra (self):
        # Initalise the dijkstra algorithm
        self.initialize()

        # Loop until not_processed is empty
        while len(self.not_processed) > 0:
            # Search unprocessed node with minimal/maximal estimate of cost
            u = self._search_min() if self.mode == "min_distance" else self._search_max() if self.mode == "max_capacity" else None

            # Update not_processed and processed
            self.not_processed.remove(u)
            self.processed.add(u)
            
            # Get successors of the current node u
            # successors(graph, u) 
            successors_u_list = successors(self.graph, u)

            # Relax all edges (u, v) adjacent to u
            for v in successors_u_list:
                # Update the estimates of the edges (u, v) adjacent to u
                if not self._filter_arc(u, v):
                    new_estimate = self._process_new_estimate(u, v)
                    # Relax the edge (u, v)
                    if self._is_better_than(new_estimate, self.path_estimates[v]):
                        self.path_estimates[v] = new_estimate
                        self.predecessors[v] = u

    
    def _backward_traverse_DAGSP_creation(self):
        # Return None if there is no path from s to t
        if (self.path_estimates[self.destination] == float('inf') and self.mode == "min_distance") or\
            (self.path_estimates[self.destination] == 0 and self.mode == "max_capacity"):
            return None
        
        # Queue Initalization
        self.visited[self.destination] = True
        queue = [self.destination]

        while len(queue) > 0:
            # Dequeue a node v
            v = queue.pop(0)

            # Return the predecessors of node v
            predecessors_v = predecessors(self.graph, v, self.predecessors_list)

            # Check the predecessors of v for shortest paths to be included in the DAG and enqueue theme if not visited
            for u in predecessors_v:
                if not self._filter_arc(u, v) and self._process_new_estimate(u, v) == self.path_estimates[v]:
                    # Pick the edge (u, v) in DAG if there is a shortest path from the source to v through u
                    add_arc(self.dagsp, u, v)
                    # If u is not visited and different from s, mark it as visited and enqueue it for later exploration
                    if not self.visited[u]:
                        self.visited[u] = True
                        if u != self.source:
                            queue.append(u)
        
        if not self.visited[self.destination] or not self.visited[self.source]:
            print("Source or destination is unvisited.")
            sys.exit()


    def construct_DAG_shortest_path (self, destination):
        # Initializations
        self.destination = destination
        self.visited = [False]*len(self.graph)
        self.dagsp = create_isolated_nodes_graph(len(self.graph), 
                                                 matrix_representation = self.matrix_representation)

        # Find the DAG pf shortest paths
        self._backward_traverse_DAGSP_creation()


    def return_path(self, destination, path_type = "first_fit", logs = None):
        if path_type not in PATH_TYPES or self.mode not in MODES:
            print("Path type or mode is inexistant.")
            sys.exit()

        if path_type == "first_fit":
            # Return the path found by the Dijkstra algorithm
            path, node = [], destination
            while node is not None:
                path = [node]+path
                node = self.predecessors[node]
            
            return path
        
        elif path_type == "random":

            def _return_random_path(graph, source, destination):
                path, node = [source], source
                while node != destination:
                    next_node = random.choice(successors(graph, node))
                    path.append(next_node)
                    node = next_node
                return path

            if self.mode == "min_distance":
                # Construct the DAG of shortest paths
                self.construct_DAG_shortest_path (destination)
                # Return a random path from the subgraph of shortest paths
                path = _return_random_path(self.dagsp, self.source, destination)
                return path
        
            elif self.mode == "max_capacity":
                # Return a random path from the subgraph of shortest paths, capacity wise
                # which unlike the name suggest "IS NOT" a DAG when 'self.mode == "max_capacity"'

                # Construct subgraph of shortest paths (fattest paths)
                self.construct_DAG_shortest_path (destination) # !!!!! LIKE SAID EARLIER, THIS IS NOT A DAG, I REPEAT THIS IS NOT A DAG !!!!!

                # Return a random path from the subgraph of fattest paths (this path my contain a cycle)
                path = _return_random_path(self.dagsp, self.source, destination)
                
                # Make the path in 'path' simple (by delecting cycles in it)
                path = make_path_simple (path)

                return path
        
        elif path_type == "random_nfilter_byshortest_path":
            # The path is chosen randomly
            # Return a random path from the subgraph of shortest paths (capacity wise or time wise depending on the mode) restricted
            # to the paths of least length (in terms of number of arcs).

            # Construct subgraph of shortest paths (fattest or fastest depending on the mode)
            self.construct_DAG_shortest_path (destination) # !!!!! LIKE SAID EARLIER, THIS IS NOT A DAG, I REPEAT THIS IS NOT A DAG !!!!!
            
            # Restrict this graph to only the shortest paths (in terms of number of arcs within the paths)
            # To do this construct a new dijkstra solver on the subgraph constructed on the precedent step
            # Set the weight matrix of the solver as the graph matrix of the previously constructed subgraph
            # Call the dijkstra algorithm on this new solver and construct the DAG of shortest path out of it
            dijkstra_solver_filter = DijkstraShortestPathsSolver(source = self.source,
                                                                 graph = self.dagsp,
                                                                 weights = self.dagsp, 
                                                                 mode = "min_distance",
                                                                 matrix_representation = self.matrix_representation)
            dijkstra_solver_filter.run_dijkstra()
            dijkstra_solver_filter.construct_DAG_shortest_path(destination)
            self.dagsp = dijkstra_solver_filter.dagsp    # And this is certainly a DAG

            # Return a random path from the DAG of shortest paths
            path = _return_random_path(self.dagsp, self.source, destination)

            return path



