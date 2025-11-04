

def depth_first_search(adj_mat, u, visited):
    visited[u] = True
    for v in range(len(adj_mat)):
        if adj_mat[u][v] == 1 and not visited[v]:
            depth_first_search(adj_mat, v, visited)


def is_connected (adj_mat, source_node = 0):
    # Intializations
    visited = [False for _ in range(len(adj_mat))]

    # Run a depth first search algorithm from the source node
    depth_first_search(adj_mat, source_node, visited)

    # Return True if the graph is connected and False otherwise
    for u in range(len(adj_mat)):
        if not visited[u]: return False
    return True


def is_strongly_connected (adj_mat):
    # Test if the nodes are accessible from all nodes
    for node in range(len(adj_mat)):
        if not is_connected (adj_mat, node): return False
    return True


if __name__ == "__main__":
    mat0 = [[0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0]]
    print("Connexity matrix 0 ", is_connected (mat0, 0))
    print("Strong Connexity matrix 0 ", is_strongly_connected (mat0))
    print("")

    mat1 = [[0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0]]
    print("Connexity matrix 1 ", is_connected (mat1, 0))
    print("Strong Connexity matrix 1 ", is_strongly_connected (mat1))
    print("")

    mat2 = [[0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0]]
    print("Connexity matrix 2 ", is_connected (mat2, 0))
    print("Strong Connexity matrix 2 ", is_strongly_connected (mat2))
