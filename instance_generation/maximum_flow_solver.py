from copy import deepcopy
import sys
import pprint

##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##### All the algorithms presenting here are guaranted to work only in the case case where the graph is a DAG 
##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#####


#################################################################  Graph traversal utilities ##########################################################
def breadth_first_search (adj_mat, source):
    # Initializations
    visited = [False]*len(adj_mat)
    pred = [None]*len(adj_mat)

    # Queue initialization
    visited[source] = True
    queue = [source]

    # Main loop
    while len(queue) > 0:
        # Dequeue a node
        node = queue.pop(0)
        # Browse through the successors of 'node' and if unvisited, update their distance from the source and their visited status
        for succ in range(len(adj_mat)):
            if adj_mat[node][succ] > 0 and not visited[succ]:
                visited[succ] = True
                pred[succ] = node
                queue.append(succ)

    return pred


def return_path(pred, end_node):
    path, node = [], end_node
    while node is not None:
        path = [node]+path
        node = pred[node]
    return path


#################################################################  Edmond-Karp Algorithm ##########################################################
class EdmondKarpSolver:

    def __init__(self, adj_mat, capacities, source, destination):
        # Set the source and the destination
        self.source = source
        self.destination = destination
        #Create the ajdacency matrix
        self.adj_mat = adj_mat
        #Create the capacity matrix 
        self.capacities = capacities
        # Create the flow matrix
        self.flow = [[0 for j in range(len(adj_mat))] for i in range(len(adj_mat))]
        # Create the residual graph
        self.residual_graph = deepcopy(capacities)


    def run_edmond_karp(self):
        # Initialiee the flow value de 0
        max_flow_val = 0

        # Execute a breath first search in the residual graph
        pred = breadth_first_search (self.residual_graph, self.source)
        
        # Augment the flow while there is an augmenting path in the resitual graph
        while pred[self.destination] is not None:
            # Return the path found by the breadth-first search
            path = return_path(pred, self.destination)

            # Process the path capacity
            path_cap = min(self.residual_graph[path[i]][path[i+1]] for i in range(len(path)-1))

            # Increase the value of the flow
            max_flow_val += path_cap

            # Update the flow and residual capacities in the path
            for i in range(len(path)-1):
                if self.adj_mat[path[i]][path[i+1]] == 1: # The case where the current arc of the path is in the original graph 
                    self.flow[path[i]][path[i+1]] += path_cap
                    self.residual_graph[path[i]][path[i+1]] = self.capacities[path[i]][path[i+1]] - self.flow[path[i]][path[i+1]]
                    self.residual_graph[path[i+1]][path[i]] = self.flow[path[i]][path[i+1]]
                
                else: # The case where the current arc of the path is not in the original graph (the reverse arc is in the original graph)
                    self.flow[path[i+1]][path[i]] -= path_cap
                    self.residual_graph[path[i+1]][path[i]] = self.capacities[path[i+1]][path[i]] - self.flow[path[i+1]][path[i]]
                    self.residual_graph[path[i]][path[i+1]] = self.flow[path[i+1]][path[i]]
            
            # Execute a breath first search in the residual graph
            pred = breadth_first_search (self.residual_graph, self.source)
        
        return max_flow_val
    

    def construct_graph_max_flow(self):
        """
        This function returns the graph where each arc is labled with the flow value of a max flow.
        This function can only be called only after a call to run_edmond_karp()
        """
        # Initialization
        max_flow_graph = [[0 for v in range(len(self.adj_mat))] for u in range(len(self.adj_mat))]

        # Label each arc of the graph with the value present in the reverse arc of its residual graph (produced by edmond-karp) 
        for u in range(len(self.adj_mat)):
            for v in range(len(self.adj_mat)):
                if self.adj_mat[u][v] == 1:
                    max_flow_graph[u][v] = self.residual_graph[v][u]
        
        return max_flow_graph


def check_flow_properties(flow_func, source, destination, flow_val, adj_mat):
    if len(adj_mat) != len(flow_func):
        print("The adjacency matrix and the flow matrix have different lengths.")
        sys.exit()
    
    for u in range(len(flow_func)):
        for v in range(len(flow_func)):
            if (adj_mat[u][v] == 1 and flow_func[u][v] == 0) or (adj_mat[u][v] == 0 and flow_func[u][v] > 0):
                print("Adjacency matrix and flow are not coherent.", u, v, adj_mat[u][v], flow_func[u][v])
                sys.exit()

    """out_source_flow = sum(flow_func[source][:]) - sum(flow_func[i][source] for i in range(len(flow_func)))
    in_destinatino_flow = sum(flow_func[i][destination] for i in range(len(flow_func))) - sum(flow_func[destination][:])
    if flow_val <= 0 or out_source_flow != flow_val or in_destinatino_flow != flow_val:
        print("Problem in flow values.", flow_val, out_source_flow, in_destinatino_flow)
        sys.exit()"""

    for u in range(len(flow_func)):
        if u != source and u != destination:
            in_flow_node = sum(flow_func[i][u] for i in range(len(flow_func)))
            out_flow_node = sum(flow_func[u][:])
            if in_flow_node != out_flow_node:
                print("Kirchhoff is not respected in node ", u)
                sys.exit()



if __name__ == "__main__":
    graph_id = 2
    if graph_id == 0:
        adj_mat = [[0, 1, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]
        
        capacities = [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]
        
        max_flow_solver = EdmondKarpSolver(adj_mat = adj_mat, 
                                           capacities = capacities, 
                                           source = 0, 
                                           destination = 3)
        max_flow_val = max_flow_solver.run_edmond_karp()
        print("The value of the max flow is ", max_flow_val)
        print("The flow ")
        pprint.pprint(max_flow_solver.flow)
        print("The residual graph")
        pprint.pprint(max_flow_solver.residual_graph)
    
    elif graph_id == 1:
        adj_mat = [[0, 1, 1, 0],
                   [0, 0, 1, 1],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]]
        
        capacities = [[0, 100, 100, 0],
                      [0, 0, 1, 100],
                      [0, 0, 0, 100],
                      [0, 0, 0, 0]]
        
        max_flow_solver = EdmondKarpSolver(adj_mat = adj_mat, 
                                           capacities = capacities, 
                                           source = 0, 
                                           destination = 3)
        max_flow_val = max_flow_solver.run_edmond_karp()
        print("The value of the max flow is ", max_flow_val)
        print("The flow ")
        pprint.pprint(max_flow_solver.flow)
        print("The residual graph")
        pprint.pprint(max_flow_solver.residual_graph)
    
    elif graph_id == 2:
        # Lien : https://iq.opengenus.org/maximum-flow-problem-overview/
        adj_mat = [[0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 1, 0],
                   [0, 1, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 0, 0]]  

        capacities = [[0, 8, 10, 0, 0, 0],
                      [0, 0, 0, 2, 7, 0],
                      [0, 3, 0, 0, 12, 0],
                      [0, 0, 0, 0, 0, 10],
                      [0, 0, 0, 4, 0, 8],
                      [0, 0, 0, 0, 0, 0]]
        
        max_flow_solver = EdmondKarpSolver(adj_mat = adj_mat, 
                                           capacities = capacities, 
                                           source = 0, 
                                           destination = 5)
        max_flow_val = max_flow_solver.run_edmond_karp()
        print("The value of the max flow is ", max_flow_val)
        print("The flow ")
        pprint.pprint(max_flow_solver.flow)
        print("The residual graph")
        pprint.pprint(max_flow_solver.residual_graph)
    