import pprint
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


######################################################  printing function   ######################################################
def print_log(logs, seed, dir_save_name, multi_flow, aggregated_flow, generated_flow_values, show = False):
    with open(dir_save_name+"output.txt", "w") as f:
        print("--------------------------------------------------------------------", file = f)
        log = logs[0]
        _, _, adj_mat, aggregated_flow = log[0], log[1], log[2], log[3]
        print("Itération 0 ", file = f)
        print("Adj matrice ", file = f)
        pprint.pprint(adj_mat, stream = f)
        print("Aggregated flow ", file = f)
        pprint.pprint(aggregated_flow, stream = f)
        print("--------------------------------------------------------------------", file = f)
        display_graph (adj_mat, aggregated_flow, "Step "+str(0), seed, dir_save_name)
        
        it = 1
        for log in logs[1:]:
            flow_capacity, path, adj_mat, aggregated_flow, weight_mat = log[0], log[1], log[2], log[3], log[4]
            print("Itération "+str(it), file = f)
            print("Path ", file = f)
            print(path, file = f)
            print("Flow capacity ", file = f)
            print(flow_capacity, file = f)
            print("Adj matrice ", file = f)
            pprint.pprint(adj_mat, stream = f)
            print("Aggregated flow ", file = f)
            pprint.pprint(aggregated_flow, stream = f)
            print("The weight matric ", file = f)
            pprint.pprint(weight_mat, stream = f)
            print("--------------------------------------------------------------------", file = f)
            display_graph (adj_mat, aggregated_flow, "Step "+str(it), seed, dir_save_name)
            it += 1
    
        print("---------------------------------- END Execution ----------------------------------", file = f)

        print("Les multiflows ", file = f)
        pprint.pprint(multi_flow, stream = f)

        print("Aggregated flow ", file = f)
        pprint.pprint(aggregated_flow, stream = f)

        print("Les valeurs de flows ", file = f)
        pprint.pprint(generated_flow_values, stream = f)

    if show:
        plt.show()


######################################################  display graph function   ######################################################
def draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def display_graph (adj_mat, weight_mat, title, seed, dir_name):
    graph = nx.DiGraph()
    edge_list = []

    for i in range(len(adj_mat)):
        for j in range(len(adj_mat)):
            if adj_mat[i][j] == 1:
                edge_list.append((i, j, {'w':weight_mat[i][j]})) 
    graph.add_edges_from(edge_list)

    fig, ax = plt.subplots()
    pos = nx.spring_layout(graph, seed = seed)

    nx.draw_networkx_nodes(graph, pos, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)

    curved_edges = [edge for edge in graph.edges() if reversed(edge) in graph.edges()]
    straight_edges = list(set(graph.edges()) - set(curved_edges))
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
    arc_rad = 0.25
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')

    edge_weights = nx.get_edge_attributes(graph, 'w')
    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
    draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False)

    plt.title(title)

    if dir_name:
        fig.savefig(dir_name+title, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

