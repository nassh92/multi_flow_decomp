import pprint

import matplotlib.pyplot as plt

import sys


def display_instance (adj_matrice, arcs, nodes, capacities, transport_times, pairs, weight_pairs, print_ = True):
    if print_:
        print("The pairs ")
        print(pairs)

        if not isinstance(weight_pairs[0], list):
            print("The weights ")
            print(weight_pairs)
        else:
            print("The degree based weights ")
            print(weight_pairs[0])

            print("The capacity based weights ")
            print(weight_pairs[1])

            print("The mincut based weights ")
            print(weight_pairs[2])

        print("Adjacency matrix ")
        #pprint.pprint(adj_matrice)
        print(adj_matrice)

        print("The capacities ")
        #pprint.pprint(capacities)
        print(capacities)

        print("Transport times ")
        #pprint.pprint([[round(e,2) for e in row] for row in transport_times])
        print([[round(e,2) for e in row] for row in transport_times])

    plt.figure()
    plt.title("The generated instance ", fontsize=15)
    for arc in arcs:
        plt.xlabel('x-axis',fontsize=15)
        plt.ylabel('y-axis',fontsize=15)
        p1, p2 = arc
        center_up = ((p1[0]+p2[0])/2 + 0.5 , (p1[1]+p2[1])/2 + 0.5)
        center_down = ((p1[0]+p2[0])/2 - 0.5 , (p1[1]+p2[1])/2 - 0.5)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
        plt.annotate(str(nodes[p1]), xy=p1)
        plt.annotate(str(nodes[p2]), xy=p2)
        plt.annotate(str(round(capacities[nodes[p1]][nodes[p2]], 2)), xy=center_up)
        plt.annotate(str(round(transport_times[nodes[p1]][nodes[p2]], 2)), xy=center_down)
        
    plt.grid()
    plt.show()