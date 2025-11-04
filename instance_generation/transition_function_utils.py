import sys


def construct_transition_functions(adj_mat, pairs):
    """
        Construct the transition functions
    """
    # Initialize the transition function
    transition_function = {(u, v):{(v,w):0 for w in range(len(adj_mat)) if adj_mat[v][w] == 1}
                                                for u in range(len(adj_mat)) for v in range(len(adj_mat))}
    # The set of all sources
    sources = {pair[0] for pair in pairs}
    # The set of all destinations
    destinations = {pair[1] for pair in pairs}

    # Initialize the transition function from the source
    transition_from_sources = {s:{(s,v):0 for v in range(len(adj_mat)) if adj_mat[s][v] == 1} for s in sources}

    # Initialize the transition function from the source
    transition_to_destinations = {d:{(u,d):0 for u in range(len(adj_mat)) if adj_mat[u][d] == 1} for d in destinations}

    return transition_function, transition_from_sources, transition_to_destinations



def update_transition_functions(ls_transition_functions, arc1, arc2, flow_amount, pairs):
    """
    
    """
    # ls_transition_functions == (transition_function, transition_from_sources, transition_to_destinations) 
    
    if arc1 is None:
        # Unpack the corresponding transition function
        transition_from_sources = ls_transition_functions[1]
        # The set of all sources
        sources = {pair[0] for pair in pairs}
        # Sanity check
        if arc2[0] not in sources:
            print("Error - can not find source.")
            sys.exit()
        # Increment the counter associated to the flow transitioning from the source to 'arc2'
        transition_from_sources[arc2[0]][arc2] += flow_amount
    
    elif arc2 is None:
        # Unpack the corresponding transition function
        transition_to_destinations = ls_transition_functions[2]
        # The set of all destinations
        destinations = {pair[1] for pair in pairs}
        # Sanity check
        if arc1[1] not in destinations:
            print("Error - can not find destination.")
            sys.exit()
        # Increment the counter associated to the flow transitioning from 'arc1' to the destination 
        transition_to_destinations[arc1[1]][arc1] += flow_amount
    
    else:
        # Unpack the corresponding transition function
        transition_function = ls_transition_functions[0]
        # Increment the counter of the flow transitioning from 'arc1' to 'arc2'
        transition_function[arc1][arc2] += flow_amount
    
    
    
