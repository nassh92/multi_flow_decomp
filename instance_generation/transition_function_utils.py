import sys
import os
sys.path.append(os.getcwd())
from utils.graph_utils import successors, get_arcs, predecessors, delete_arc


def construct_transition_functions(graph, pairs, predecessors_list = None):
    """
        Construct the transition functions
    """
    # Initialize the transition function
    transition_function = {(u, v):{(v,w):0 for w in successors(graph, v)}
                                                for u, v in get_arcs(graph)}
    # The set of all sources
    sources = {pair[0] for pair in pairs}
    # The set of all destinations
    destinations = {pair[1] for pair in pairs}

    # Initialize the transition function from the source
    transition_from_sources = {s:{(s,v):0 for v in successors(graph, s)} for s in sources}

    # Initialize the transition function from the source
    transition_to_destinations = {d:{(u,d):0 for u in predecessors(graph, d, predecessors_list)} 
                                                for d in destinations}

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
    
    
    
