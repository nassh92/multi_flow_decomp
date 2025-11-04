from copy import deepcopy

import random

import sys

import pprint

from test_graph_utils import test_sg_coherence


def print_log(logs):
    print("--------------------------------------------------------------------")
    log = logs[0]
    _, _, adj_mat, aggregated_flow = log[0], log[1], log[2], log[3]
    print("Adj matrice ")
    pprint.pprint(adj_mat)
    print("Aggregated flow ")
    pprint.pprint(aggregated_flow)
    print("--------------------------------------------------------------------")

    for log in logs[1:]:
        flow_capacity, path, adj_mat, aggregated_flow = log[0], log[1], log[2], log[3]
        print("Path ")
        print(path)
        print("Flow capacity ")
        print(flow_capacity)
        print("Adj matrice ")
        pprint.pprint(adj_mat)
        print("Aggregated flow ")
        pprint.pprint(aggregated_flow)
        print("--------------------------------------------------------------------")


def construct_path_flow_decomp (capacities, source, destination, dg_sp, logs = None, old_path = None, og_dag = None,
                                path_estimates = None):
    """
    This algorithms finds a path based on the flow decomposition algorithm and works only if 'capacities' defines a flow in the flow in
    the subgraph 'dg_sp'
    """
    cycles = []
    # Initialize the path and the current node
    path, node = [source], source
    while node != destination:
        # Chose randomly the next node to be added to the path
        try: 
            next_node = random.choice([v for v in range(len(dg_sp[node])) if dg_sp[node][v]==1])
        except:
            print("The node ", node)
            print("List all succ ", dg_sp[node])
            print("List succs ", [v for v in range(len(dg_sp[node])) if dg_sp[node][v]==1])
            print("List succs OG DAG ", og_dag[node])
            if old_path is not None: 
                path_weight = []
                for i in range(len(old_path)-1):
                    path_weight.append(capacities[old_path[i]][old_path[i+1]])
                print("Path weight after", path_weight)
                if len(cycles) > 0:
                    print("Last cycle ", cycles[-1])
                else:
                    print("No cycle.")
            sys.exit()

        # If the chosen not is not in 'path', add it
        if next_node not in path:
            path.append(next_node)
            node = next_node
        else:
            # If the chosen is in 'path', this means that adding it would create a cycle.
            # In this cas :
            # - Delete all the nodes of this cycle which are in 'path'.
            # - Decrease the capacities of the arcs of this cycles by how much capacity it holds and delete the arcs with capacity 0. 
            # - An then, we start from over from the 'source' to find a new random path.
            # Each time a cycle is found an arc is deleted, so the algorithm terminates (in o(m))

            # Deleting the nodes of the cycle from 'path'
            cur_cyc_node = path[-1]
            cap_cycle, cycle = capacities[cur_cyc_node][next_node], [next_node]
            while cur_cyc_node != next_node: 
                cur_cyc_node = path.pop()
                cap_cycle = min(cap_cycle, capacities[cur_cyc_node][cycle[0]])
                cycle = [cur_cyc_node]+cycle
            
            # Deducing the capacity of the cycle (cap_cycle), from the capacities of the graph. and updating 'dg_sp' accordingly
            for k in range(len(cycle)-1):
                capacities[cycle[k]][cycle[k+1]] -= cap_cycle
                if capacities[cycle[k]][cycle[k+1]] == 0: dg_sp[cycle[k]][cycle[k+1]]=0
            
            cycles.append(deepcopy(cycle))
            
            # Test
            test_res = test_sg_coherence(adj_mat=dg_sp, source=source, destination=destination)
            if len(test_res[0]) != 0 or len(test_res[1]) != 0:
                print("The subgraph of shortest path is not coherent. Call 3", path_estimates[destination])
                print("source+dest - conn_nodes", test_res[0])
                print("conn_nodes - source+dest", test_res[1])
                print("---------------------------")
                if old_path is not None: 
                    path_weight = []
                    for i in range(len(old_path)-1):
                        path_weight.append(capacities[old_path[i]][old_path[i+1]])
                    print("Path weight after", path_weight)
                    if len(cycles) > 0:
                        print("Last cycle ", cycles[-1])
                    else:
                        print("No cycle.")

                sys.exit()

            # Initialize the path and the current node (start from over)
            path, node = [source], source
    
    return path

            