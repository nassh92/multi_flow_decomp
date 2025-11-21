import sys
import os
import numpy as np
from copy import deepcopy
import math
sys.path.append(os.getcwd())
from instance_generation.generic_multi_flow_instances_generator import generate_multi_flow_instance, fetch_ajust_ncorrect_flow_network_data
from instance_generation.demand_generator import process_demand_weights_distribution, process_desired_flow_values
from msmd.multi_flow_desag_instance_utils import MultiFlowDesagInstance


MAXIMAL_CONGESTION_RATE = 1.0


def construct_simon_instances(path_network_data,
                              path_maximal_flow_val_data,
                              dir_save_name,
                              start_time = 6,
                              end_time = 20,
                              time_period = 25,
                              locs = (8, 12.5, 17),
                              min_fl = 1,
                              nb_it_print = None):
    # Fetch the graph data (empty network, structural properties) from the numpy file
    dict_empty_network = np.load(path_network_data, allow_pickle = True).flat[0]
    graph = dict_empty_network["graph"]
    capacities = dict_empty_network["capacities"]
    raw_transport_times = dict_empty_network["transport_times"]
    original_total_pairs = dict_empty_network["pairs"]
    weight_pairs = dict_empty_network["weight_pairs"]

    # Fetch dict instances (serving to bound the flow value generated)
    dict_instances_max_flow = np.load(path_maximal_flow_val_data, allow_pickle = True).flat[0]

    # The time line
    time_line = np.linspace(start_time, end_time, (end_time - start_time) * 100 + 1)
    weights_demand = process_demand_weights_distribution(time_line, locs = locs)
    
    # Generate the real instances
    for num_instance, _, _ in dict_instances_max_flow:
        # Get the maximal flow values
        print("Treatment of instance ", num_instance)
        mfd_instance = dict_instances_max_flow[(num_instance, _, _)][0]
        all_pairs = deepcopy(mfd_instance.pairs)
        maximal_flow_values = deepcopy(mfd_instance.original_flow_values)
        print(maximal_flow_values)
        """
        Generate flow values for each period
        Generate data for each period
        """
        for ind_time in range(len(time_line)):
            time_diff = round((time_line[ind_time] - math.floor(time_line[ind_time])) * 100)
            if time_diff % time_period == 0:
                # Process the desired flow values
                all_desired_flow_values = process_desired_flow_values(weights_demand[ind_time],
                                                                    maximal_congestion_rate = MAXIMAL_CONGESTION_RATE,
                                                                    maximal_flow_values = maximal_flow_values)
                
                # Process the desired flow values
                restricted_weight_pairs = [weight_pairs[i] for i in range(len(weight_pairs)) if original_total_pairs[i] in all_pairs]
                return_multi_flow_dict = generate_multi_flow_instance(deepcopy(graph), 
                                                                    deepcopy(capacities), 
                                                                    deepcopy(raw_transport_times), 
                                                                    deepcopy(all_pairs), 
                                                                    all_desired_flow_values, 
                                                                    min_fl = min_fl, 
                                                                    nb_it_print = nb_it_print,
                                                                    weight_pairs = restricted_weight_pairs,
                                                                    return_transition_function = True)
                
                # Correct/ajust the network data after flow generation
                return_dict_ajusted_data = fetch_ajust_ncorrect_flow_network_data(graph,
                                                                            raw_transport_times, 
                                                                            all_pairs, 
                                                                            return_multi_flow_dict)
                
                # Saving the file
                np.save(os.path.join(dir_save_name, 
                                    "multi_flow_instance_"+str(num_instance)+"_"+str(round(time_line[ind_time], 2))), 
                        {"pairs":return_dict_ajusted_data["pairs"],
                         "flow_values":return_dict_ajusted_data["flow_values"],
                         "multi_flow":return_dict_ajusted_data["multi_flow"]})



def main():
    construct_simon_instances(path_network_data = "data/real_data/pre_processed/LieuSaint/real_instance_lieusaint.npy",
                              path_maximal_flow_val_data = "data/real_data/pre_processed/LieuSaint/data_instances.npy",
                              dir_save_name = "data/data_simon/instances/",
                              start_time = 6,
                              end_time = 20,
                              time_period = 25,
                              locs = (8, 12.5, 17),
                              min_fl = 1,
                              nb_it_print = None)


if __name__ == "__main__":
    main()