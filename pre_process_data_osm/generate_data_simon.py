import sys
import os
import numpy as np
from copy import deepcopy

sys.path.append(os.getcwd())
from instance_generation.generic_multi_flow_instances_generator import generate_multi_flow_instance, fetch_ajust_ncorrect_flow_network_data
from instance_generation.demand_generator import 
from pre_process_data_osm.osm_to_mfd import construct_real_instances
from msmd.multi_flow_desag_instance_utils import MultiFlowDesagInstance



def construct_simon_instances(path_network_data,
                              path_maximal_flow_val_data,
                              dir_save_name_multiflow,
                              dir_save_name_mfd,
                              nb_instances,
                              suffix_fname,
                              car_size = 1,
                              min_fl = 1,
                              nb_max_draws_pairs = 10,
                              nb_it_print = None,
                              save_dir = None,
                              matrix_representation = True,
                              generate_figure = None):
    # Fetch the graph data from the numpy file
    dict_empty_network = np.load(path_network_data, allow_pickle = True).flatten[0]
    graph = dict_empty_network["graph"]
    capacities = dict_empty_network["capacities"]
    raw_transport_times = dict_empty_network["transport_times"]
    all_pairs = dict_empty_network["pairs"]
    weight_pairs = dict_empty_network["weight_pairs"]

    # Fetch dict instances
    dict_instances_max_flow = np.load(path_maximal_flow_val_data, allow_pickle = True).flatten[0]
    
    # Generate the real instances
    dict_instances = {}
    for num_instance, _, _ in dict_instances_max_flow:
        # Get the maximal flow values
        mfd_instance = dict_instances_max_flow[(num_instance, _, _)][0]
        all_pairs = deepcopy(mfd_instance.pairs)
        all_desired_flow_values = deepcopy(mfd_instance.original_flow_values)

        """
        Generate flow values for each period
        Generate data for each period
        """

        return_multi_flow_dict = generate_multi_flow_instance(deepcopy(graph), 
                                                              deepcopy(capacities), 
                                                              deepcopy(raw_transport_times), 
                                                              all_pairs, 
                                                              all_desired_flow_values, 
                                                              min_fl = min_fl, 
                                                              nb_it_print = nb_it_print,
                                                              weight_pairs = deepcopy(weight_pairs),
                                                              return_transition_function = True)
        
        # Correct/ajust the network data after flow generation
        return_dict_ajusted_data = fetch_ajust_ncorrect_flow_network_data(graph,
                                                                     raw_transport_times, 
                                                                     all_pairs, 
                                                                     all_desired_flow_values,
                                                                     return_multi_flow_dict)

        # Construct mfd_instance
        mfd_instance = MultiFlowDesagInstance(deepcopy(return_dict_ajusted_data["corr_graph"]), 
                                            deepcopy(return_dict_ajusted_data["aggregated_flow"]),
                                            deepcopy(return_dict_ajusted_data["transport_times"]), 
                                            deepcopy(return_dict_ajusted_data["pairs"]), 
                                            deepcopy(return_dict_ajusted_data["flow_values"]), 
                                            deepcopy(return_dict_ajusted_data["ls_transition_function"]),
                                            update_transport_time = True,
                                            update_transition_functions = True)
        # Saving the file
        np.save(os.path.join(dir_save_name_multiflow, 
                             "multi_flow_instance_"+str(num_instance)), 
                {"pairs":deepcopy(return_dict_ajusted_data["pairs"]),
                 "flow_values":deepcopy(return_dict_ajusted_data["flow_values"]),
                 "multi_flow":return_dict_ajusted_data["multi_flow"]})
        dict_instances[(num_instance, True, True)] = (mfd_instance, return_dict_ajusted_data["multi_flow"])

        # Saving the file
        np.save(os.path.join(dir_save_name_multiflow, 
                             "multi_flow_instance_"+str(num_instance)), 
                {"pairs":deepcopy(return_dict_ajusted_data["pairs"]),
                 "flow_values":deepcopy(return_dict_ajusted_data["flow_values"]),
                 "multi_flow":return_dict_ajusted_data["multi_flow"]})
        dict_instances[(num_instance, True, True)] = (mfd_instance, return_dict_ajusted_data["multi_flow"])




def main():
    pass


if __name__ == "__main__":
    main()