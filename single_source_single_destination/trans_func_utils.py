

def construct_tree_arcbfs (source, 
                           adj_mat, 
                           trans_func, 
                           trans_from_sources, 
                           subg_filterer,
                           max_trans_func_successor):
    """
        !!! max_trans_func_successor must be set to False if we use RLbased desaggregation algorithms !!!
    """
    # Initalizations, create and push to the queue the outgoing arcs from 'source' 
    queue = []
    subgraph_mat = [[0 for j in range(len(adj_mat))] for i in range(len(adj_mat))]
    visited = [[False for j in range(len(adj_mat))] for i in range(len(adj_mat))]
    max_val_flow_arc = -1 if not max_trans_func_successor else max(trans_from_sources[source][succ_arc] for succ_arc in trans_from_sources[source])
    for succ_arc in trans_from_sources[source]:
        if adj_mat[succ_arc[0]][succ_arc[1]] == 1 and\
            trans_from_sources[source][succ_arc] > 0 and\
                trans_from_sources[source][succ_arc] >= max_val_flow_arc and\
                    subg_filterer.check_filtering_cond_successors (source, 
                                                                   succ_arc[1], 
                                                                   first_check = True): 
            queue.append((succ_arc[0], succ_arc[1]))
            subgraph_mat[succ_arc[0]][succ_arc[1]] = 1
            visited[succ_arc[0]][succ_arc[1]] = True

    # Loop as long as queue is not empty
    while queue:
        # Pop an arc from queue
        arc = queue.pop(0)

        # Check successors of 'arc' and push them if unvisited
        max_val_flow_arc = -1 if not max_trans_func_successor else max(trans_func[arc][succ_arc] for succ_arc in trans_func[arc])
        for succ_arc in trans_func[arc]:
            if adj_mat[succ_arc[0]][succ_arc[1]] == 1 and\
                trans_func[arc][succ_arc] > 0 and\
                    trans_func[arc][succ_arc] >=  max_val_flow_arc and\
                        not visited[succ_arc[0]][succ_arc[1]] and\
                            subg_filterer.check_filtering_cond_successors (arc, 
                                                                           succ_arc[1]):
                queue.append((succ_arc[0], succ_arc[1]))
                subgraph_mat[succ_arc[0]][succ_arc[1]] = 1
                visited[succ_arc[0]][succ_arc[1]] = True

    return subgraph_mat


def construct_antitree_arcbfs (destination, 
                               adj_mat, 
                               trans_func, 
                               trans_to_destinations,
                               subg_filterer):
    # Initalizations, create and push to the queue the outgoing arcs from 'source' 
    queue = []
    subgraph_mat = [[0 for j in range(len(adj_mat))] for i in range(len(adj_mat))]
    visited = [[False for j in range(len(adj_mat))] for i in range(len(adj_mat))]
    for pred_arc in trans_to_destinations[destination]:
        if adj_mat[pred_arc[0]][pred_arc[1]] == 1 and\
                trans_to_destinations[destination][pred_arc] > 0: 
            queue.append((pred_arc[0], pred_arc[1]))
            subgraph_mat[pred_arc[0]][pred_arc[1]] = 1
            visited[pred_arc[0]][pred_arc[1]] = True

    # Loop as long as the queue is not empty
    while queue:
        # Pop an arc from queue
        arc = queue.pop(0)

        # Construct predecessors of arc
        predecessors = [pred_arc for pred_arc in trans_func if arc in trans_func[pred_arc] and\
                                                            trans_func[pred_arc][arc] > 0 and\
                                                            adj_mat[pred_arc[0]][pred_arc[1]] == 1 and\
                                                            not visited[pred_arc[0]][pred_arc[1]] and\
                                                            subg_filterer.check_filtering_cond_predecessors (arc, 
                                                                                                             pred_arc[0])]

        # Check successors of 'arc' and push them if unvisited
        for pred_arc in predecessors:
            queue.append((pred_arc[0], pred_arc[1]))
            subgraph_mat[pred_arc[0]][pred_arc[1]] = 1
            visited[pred_arc[0]][pred_arc[1]] = True

    return subgraph_mat


def subgraph_source_dest (source, 
                          destination, 
                          adj_mat, 
                          trans_func, 
                          trans_from_sources, 
                          trans_to_destinations, 
                          subg_filterer,
                          max_trans_func_successor = False):
    return construct_antitree_arcbfs (destination, 
                                      construct_tree_arcbfs (source, 
                                                             adj_mat, 
                                                             trans_func, 
                                                             trans_from_sources,
                                                             subg_filterer,
                                                             max_trans_func_successor), 
                                      trans_func, 
                                      trans_to_destinations,
                                      subg_filterer)