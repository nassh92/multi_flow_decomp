def all_accessible_from_source(adj_mat, source):
    # Initializations
    accs = {source}
    visited = [False for _ in range(len(adj_mat))]
    visited[source] = True
    
    # Create and push queue
    queue = [source]

    # Loop as long as queue is not empty
    while queue:
        # Pop from queue
        node = queue.pop(0)

        # Check successors of 'node' and push them if unvisited
        for next_node in range(len(adj_mat)):
            if adj_mat[node][next_node] == 1 and not visited[next_node]:
                accs.add(next_node)
                visited[next_node] = True
                queue.append(next_node)
    
    return accs


def all_accessible_to_destination(adj_mat, destination):
    # Initializations
    accs = {destination}
    visited = [False for _ in range(len(adj_mat))]
    visited[destination] = True
    
    # Create and push queue
    queue = [destination]

    # Loop as long as queue is not empty
    while queue:
        # Pop from queue
        node = queue.pop(0)

        # Check successors of 'node' and push them if unvisited
        for pred_node in range(len(adj_mat)):
            if adj_mat[pred_node][node] == 1 and not visited[pred_node]:
                accs.add(pred_node)
                visited[pred_node] = True
                queue.append(pred_node)

    return accs


def return_conn_nodes(adj_mat):
    out_degrees = [sum(adj_mat[i][:]) for i in range(len(adj_mat))]
    in_degrees = [sum(adj_mat[i][j] for i in range(len(adj_mat))) for j in range(len(adj_mat))]
    set_conn_nodes = {i for i in range(len(adj_mat)) if out_degrees[i] >= 1 or in_degrees[i] >= 1}
    return set_conn_nodes


def test_sg_coherence(adj_mat, source, destination):
    from_source = all_accessible_from_source(adj_mat, source)
    to_destination = all_accessible_to_destination(adj_mat, destination)
    set_conn_nodes = return_conn_nodes(adj_mat)
    return [from_source & to_destination - set_conn_nodes, set_conn_nodes - from_source & to_destination]