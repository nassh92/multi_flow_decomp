import sys



########################################################################################################################
################################################ GRAPH API : adjancency list/matrix  ###################################
########################################################################################################################

def create_isolated_nodes_graph(size, matrix_representation = True):
     if matrix_representation:
          return [[0 for j in range(size)] for i in range(size)]
     else:
          return {i:[] for i in range(size)}
     

def has_arc(adjacency, u, v):
     if isinstance(adjacency, list):
          return adjacency[u][v] == 1

     elif isinstance(adjacency, dict):
          if u not in adjacency:
               print("The node 'u' is not in the graph.")
               sys.exit()
          else:
               return v in adjacency[u]
          

def add_arc(adjacency, u, v):
     if isinstance(adjacency, list):
          adjacency[u][v] = 1
     
     elif isinstance(adjacency, dict):
          if u not in adjacency:
               print("The node 'u' is not in the graph.")
               sys.exit()
          else:
               adjacency


def delete_arc(adjacency, u, v):
     if isinstance(adjacency, list):
          adjacency[u][v] = 0
     
     elif isinstance(adjacency, dict):
          if u not in adjacency:
               print("The node 'u' is not in the graph.")
               sys.exit()
          else:
               del adjacency[u][v]


def successors(adjacency, u):
     if isinstance(adjacency, list):
          return [v for v in range(len(adjacency)) if adjacency[u][v] == 1]
     
     elif isinstance(adjacency, dict):
          if u not in adjacency:
               print("The node 'u' is not in the graph.")
               sys.exit()
          else:
               return [v for v in adjacency[u]]
          

def predecessors(adjacency, v, predecessors_list = None):
     if predecessors_list is None:
          if isinstance(adjacency, list):
               for u in range (len(adjacency)):
                    if adjacency[u][v] == 1:
                         predecessors_list.append(u)
          
          elif isinstance(adjacency, dict):
               for u in adjacency:
                    if v in adjacency[u]:
                         predecessors_list.append(u)
     else:
          return predecessors_list[v]
          



########################################################################################################################
########################################### SOME FUNCTIONS FOR GRAPH ALGORITHMS ########################################
########################################################################################################################

def construct_tree_bsf (adj_mat, source):
    # Initializations
    tree = [[0 for _ in range(len(adj_mat))] for _ in range(len(adj_mat))]
    visited = [False] * len(adj_mat)
    visited[source] = True
    queue = [source]

    # Main loop
    while queue:
        # Pop the head of the queue
         node = queue.pop(0)

         # Browse the successors of 'node' and treat them if they are unvisited
         for succ in range(len(adj_mat)):
              if adj_mat[node][succ] == 1 and not visited[succ]:
                   visited[succ] = True
                   tree[node][succ] = 1
                   queue.append(succ)
    
    return tree


def construct_anti_tree_bfs (adj_mat, destination):
     # Initializations
    anti_tree = [[0 for _ in range(len(adj_mat))] for _ in range(len(adj_mat))]
    visited = [False] * len(adj_mat)
    visited[destination] = True
    queue = [destination]

    # Main loop
    while queue:
         # Pop the head of the queue
         node = queue.pop(0)
         
         # Browse the successors of 'node' and treat them if they are unvisited
         for pred in range(len(adj_mat)):
             if adj_mat[pred][node] == 1 and not visited[pred]:
                  visited[pred] = True
                  anti_tree[pred][node] = 1
                  queue.append(pred)

    return anti_tree


"""def construct_DAG_arc_shortest_path (adj_mat, source, destination):
     # Construct tree rooted in source
     tree = construct_tree_bsf (adj_mat, source)
     # Construct anti tree rooted in destination
     anti_tree = construct_anti_tree_bfs (adj_mat, destination)
     # Take the arcs found in the tree rooted in srouce and the antitree rooted in destination 
     dag_arc_shortest_path = construct_anti_tree_bfs (tree, destination)
     return dag_arc_shortest_path"""


def adjacency_matrix_union (adj_mat1, adj_mat2):
    adj_mat_inters = [[0 for _ in range(len(adj_mat1))] for _ in range(len(adj_mat1))]
    for i in range(len(adj_mat1)):
         for j in range(len(adj_mat1)):
              if adj_mat1[i][j] == 1 or adj_mat2[i][j] == 1:
                  adj_mat_inters[i][j] = 1
    return adj_mat_inters


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
