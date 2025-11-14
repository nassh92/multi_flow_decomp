import sys


########################################################################################################################
################################################ GRAPH API : adjancency list/matrix  ###################################
########################################################################################################################

def create_isolated_nodes_graph(size, matrix_representation = True):
     if matrix_representation:
          return [[0 for v in range(size)] for u in range(size)]
     else:
          return {u:[] for u in range(size)}


def is_adjacency_matrix(graph):
     if isinstance(graph, list):
          return True
     elif isinstance(graph, dict):
          return False
     else:
          print('Adjacency not recognize.')
          sys.exit()


def init_graph_arc_attribute_vals(graph, init_val = 0):
     if isinstance(graph, list):
          return [[init_val for v in range(len(graph))]
                              for u in range(len(graph))]
     elif isinstance(graph, dict):
          return {u:{v:init_val for v in successors(graph, u)} for u in graph}
     

def get_nodes(graph):
     if isinstance(graph, list):
          return [u for u in range(len(graph))]
     
     elif isinstance(graph, dict):
          return list(graph.keys())
     

def get_arcs(graph):
     if isinstance(graph, list):
          return [(u, v) for u in range(len(graph)) for v in range(len(graph)) if graph[u][v] == 1]
     
     elif isinstance(graph, dict):
          return [(u, v) for u in graph for v in graph[u]]


def has_arc(graph, u, v):
     if isinstance(graph, list):
          return graph[u][v] == 1

     elif isinstance(graph, dict):
          if u not in graph:
               print("Func - has_arc")
               print("The node 'u' is not in the graph.")
               sys.exit()
          else:
               return v in graph[u]
          

def add_arc(graph, u, v):
     if isinstance(graph, list):
          graph[u][v] = 1
     
     elif isinstance(graph, dict):
          if u not in graph:
               print("Func - add_arc")
               print("The node 'u' is not in the graph.")
               sys.exit()
          else:
               graph[u].append(v)


def delete_arc(graph, u, v):
     if isinstance(graph, list):
          graph[u][v] = 0
     
     elif isinstance(graph, dict):
          if u not in graph:
               print("Func - delete_arc")
               print("The node 'u' is not in the graph.")
               sys.exit()
          else:
               graph[u].remove(v)


def successors(graph, u):
     if isinstance(graph, list):
          return [v for v in range(len(graph)) if graph[u][v] == 1]
     
     elif isinstance(graph, dict):
          if u not in graph:
               print("Func - successors")
               print("The node 'u' is not in the graph.")
               sys.exit()
          else:
               return graph[u]          
          

def predecessors(graph, v, predecessors_list = None):
     if predecessors_list is None:
          predecessors_v = []
          if isinstance(graph, list):
               for u in range (len(graph)):
                    if graph[u][v] == 1:
                         predecessors_v.append(u)

          elif isinstance(graph, dict):
               for u in graph:
                    if v in graph[u]:
                         predecessors_v.append(u)
          return predecessors_v
     else:
          return predecessors_list[v]


def construct_predecessors_list(graph):
     predecessors_list = {}
     for v in get_nodes(graph): predecessors_list[v] = predecessors(graph, v) 
     return predecessors_list


def out_degree(graph, u):
     return len(successors(graph, u))
          

def in_degree(graph, v, predecessors_list = None):
     return len(predecessors(graph, v, predecessors_list = predecessors_list))


def sum_out_attributes(arc_attribute_vals, graph, u):
     successors_u_list = successors(graph, u)
     return sum(arc_attribute_vals[u][v] for v in successors_u_list)


def sum_in_attributes(arc_attribute_vals, graph, v, predecessors_list = None):
     predecessors_v_list = predecessors(graph, v, predecessors_list = predecessors_list)
     return sum(arc_attribute_vals[u][v] for u in predecessors_v_list)



########################################################################################################################
########################################### SOME FUNCTIONS FOR GRAPH ALGORITHMS ########################################
########################################################################################################################

def construct_tree_bsf (graph, 
                        source,
                        matrix_representation = True):
    # Initializations
    tree = create_isolated_nodes_graph(len(graph), 
                                       matrix_representation = matrix_representation)
    visited = [False] * len(graph)
    visited[source] = True
    queue = [source]

    # Main loop
    while queue:
        # Pop the head of the queue
         node = queue.pop(0)

         # Browse the successors of 'node' and treat them if they are unvisited
         for succ in range(len(graph)):
              if has_arc(graph, node, succ) and not visited[succ]:
                   visited[succ] = True
                   add_arc(tree, node, succ)
                   queue.append(succ)
    
    return tree


def construct_anti_tree_bfs (graph, destination, matrix_representation = True):
     # Initializations
    anti_tree = create_isolated_nodes_graph(len(graph), 
                                            matrix_representation = matrix_representation)
    visited = [False] * len(graph)
    visited[destination] = True
    queue = [destination]

    # Main loop
    while queue:
         # Pop the head of the queue
         node = queue.pop(0)
         
         # Browse the successors of 'node' and treat them if they are unvisited
         for pred in range(len(graph)):
             if has_arc(graph, pred, node) and not visited[pred]:
                  visited[pred] = True
                  add_arc(anti_tree, pred, node)
                  queue.append(pred)

    return anti_tree


"""def construct_DAG_arc_shortest_path (adj_mat, source, destination):
     # Construct tree rooted in source
     tree = construct_tree_bsf (adj_mat, source)
     # Construct anti tree rooted in destination
     anti_tree = construct_anti_tree_bfs (adj_mat, destination)
     # Take the arcs found in the tree rooted in srouce and the antitree rooted in destination 
     dag_arc_shortest_path = construct_anti_tree_bfs (tree, destination)
     return dag_arc_shortest_path
"""


def graph_union (graph1, 
                     graph2,
                     matrix_representation = True):
    """
    Returns the union of two graphs represented by their graph matrices/lists over
    the same nodes.
    """
    arcs = set(get_arcs(graph1)) | set(get_arcs(graph2))
    adj_inters = create_isolated_nodes_graph(len(graph1), 
                                             matrix_representation = matrix_representation)
    for arc in arcs: add_arc(adj_inters, arc[0], arc[1])
    return adj_inters


def make_path_simple (path):
     simple_path = []
     for i in range(len(path)):
          # Search the i'th node of 'path' in the 'simple_path' (the simple path we construct)
          j = 0
          while j < len(simple_path) and simple_path[j] != path[i]: j += 1
          
          # If the i'th node is not found on 'simple_path' append the node else ignore all nodes after it in 'simple_path'
          if j == len(simple_path):
               simple_path.append(path[i])
          else:
               simple_path = simple_path[:j+1]
     
     return simple_path


def check_arc_inclusion (arc, path):
     return any(path[i] == arc[0] and path[i+1] == arc[1] for i in range(len(path) - 1))


def path_intersection(path1, path2):
     i, j, intersection = 0, 0, []
     while i < len(path1) - 1:
          while j < len(path2) - 1 and (path1[i], path1[i+1]) != (path2[j], path2[j+1]): j += 1
          if j < len(path2) - 1: intersection.append((path1[i], path1[i+1]))
          i += 1
          j = 0
     return intersection


def group_path_intersection (group_path, path):
     group_intersection = []
     for path1 in group_path:
          group_intersection.append(path_intersection(path1, path))
     return group_intersection


class Node:
    
    def __init__(self, data, key, children = None, parent = None):
        """
        data : data (any type)
        children : children list
        parent : Noe type
        """
        self.data = data
        self.key = key
        self.children = children
        self.parent = parent   
    
    def get_data(self):
        return self.data   
    
    def get_key(self):
        return self.key
    
    def set_key(self, key):
        self.key = key
    
    def add_child(self, node):
        self.children.append(node) 
        
    def get_children(self):
        return self.children
        
    
    def set_parent(self, parent):
        self.parent = parent
        
    def get_parent(self):
        return self.parent
    


if __name__ == "__main__":
     path1 = [1, 2, 3, 4, 5, 6, 7, 8]
     path2 = [1, 2, 3, 10, 11, 5, 6, 7, 8]
