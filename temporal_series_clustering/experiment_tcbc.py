import json
import string
import time

import numpy as np

from temporal_series_clustering.cluster.algorithms.tcbc import TCBC
from temporal_series_clustering.cluster.cluster_utils import store_clusters_json
from temporal_series_clustering.patterns.experiment_patterns import *

from temporal_series_clustering.sheaf.sheaf_model import create_sheaf_model
from temporal_series_clustering.sheaf.utils import propagate_sheaf_values
from temporal_series_clustering.static.constants import ITEMS_PER_DAY
from temporal_series_clustering.storage.clusters_history import ClustersHistory
from temporal_series_clustering.storage.consistencies import ConsistenciesHistory
from temporal_series_clustering.storage.epsilons import EpsilonValues
from temporal_series_clustering.storage.simplified_graphs import SimplifiedGraphsHistory


def remove_key(d, key):
    if key in d:
        del d[key]
    for k, v in d.items():
        if isinstance(v, dict):
            remove_key(v, key)
    return d


def perform_epsilon_optimization(consistencies_history: ConsistenciesHistory, base_vertices, use_historical: bool,
                                 temporal_window, temporal_offset, values_history):
    # Define epsilon values object
    epsilon_values = EpsilonValues()

    # Create the simplified graphs
    simplified_graphs = SimplifiedGraphsHistory(num_base_vertices=len(all_nodes))

    simplified_graphs.create_simplified_graphs(consistencies_history=consistencies_history)

    total_times = {}

    # Append use historical sufix on files
    sufix = "no_" if not use_historical else ""

    # Perform clustering per each possible epsilon
    for epsilon in np.arange(0.00, 0.30, 0.01):
        epsilon = round(epsilon, 2)
        start_time = time.time()
        tcbc = TCBC(consistencies_history=consistencies_history,
                    clusters_history=ClustersHistory(),
                    simplified_graphs_history=simplified_graphs,
                    all_nodes=base_vertices,
                    epsilon=epsilon,
                    use_historical=use_historical,
                    temporal_window=temporal_window,
                    temporal_offset=temporal_offset,
                    values_history=values_history)

        historical_info_used = tcbc.perform_all_instants_clustering()

        epsilon_values.insert_all_info_on_epsilon(epsilon=epsilon, historical_info_used=historical_info_used,
                                                  clusters_history=tcbc.clusters_history)

        total_times[epsilon] = time.time() - start_time
        print(f"On epsilon {epsilon}, lasted time {time.time() - start_time}")

        epsilon_values_filtered = remove_key(epsilon_values.info, 'overlapping_clusters')

        # Store epsilon file
        store_clusters_json(epsilon_values_filtered,
                            f'../results/experiment_{len(all_nodes)}_sources_{sufix}historical.json')

        with open(f'../results/experiment_{len(all_nodes)}_sources_times_{sufix}historical.json', 'w') as f:
            json.dump(total_times, f)

    return epsilon_values


def generate_characters(n):
    """
    Generate ASCII uppercase characters, so when the limit is reached, combinations are provided (e.g. AA)
    :return:
    """
    ascii_chars = string.ascii_uppercase
    length = len(ascii_chars)
    result = []
    for i in range(n):
        if i < length:
            result.append(ascii_chars[i])
        else:
            quotient, remainder = divmod(i, length)
            result.append(ascii_chars[quotient - 1] + ascii_chars[remainder])
    return result


if __name__ == "__main__":
    """ Six original sources
    num_patterns = 6

    # Generate patterns for each possible weekday
    weekday_patterns = generate_patterns(weekday="weekday", total_num=num_patterns)
    saturday_patterns = generate_patterns(weekday="saturday", total_num=num_patterns)
    sunday_patterns = generate_patterns(weekday="sunday", total_num=num_patterns)

    # Define patterns to concat all the weekday patterns
    patterns = []
    # Iterate over each pattern (same length)
    for i in range(len(weekday_patterns)):
        # Concat the three patterns
        pattern_concat = weekday_patterns[i] + saturday_patterns[i] + sunday_patterns[i]
        # Append to list of patterns
        patterns.append(pattern_concat)
    """
    # With the seventh pattern as combination of best epsilon clusters
    num_patterns = 6
    # Temporal window and offset
    temporal_window = 5
    temporal_offset = 24

    patterns = generate_weeks_base_patterns(total_weeks=6)

    # Add little noise to each one of the patterns
    for i in range(len(patterns)):
        patterns[i] = add_instant_variation(patterns[i], noise_level=0.02)

    # Define a list indicating the instants on which use historical. Empty for no use
    use_historical = []
    # All instants except
    # use_historical = [i for i in range(len(patterns[0]))]

    # Get base vertices
    all_nodes = generate_characters(num_patterns)
    # Create sheaf model with base vertices
    sheaf_model = create_sheaf_model(all_nodes)

    # For testing purposes we are going to generate the output for each predictor
    reading_freq = 1

    summary, filtration, mean, values = propagate_sheaf_values(sheaf=sheaf_model,
                                                               data_sources_output=patterns,
                                                               reading_freq=reading_freq,
                                                               pattern_length=len(patterns[0]),
                                                               base_vertices=all_nodes)

    # Store the actual values of each pattern for each instant as the filtration
    source_values = {i: {all_nodes[j]: patterns[j][i] for j in range(len(all_nodes))} for i in range(len(patterns[0]))}

    # Create the different storages
    consistencies_history = ConsistenciesHistory(filtration)
    values_history = ConsistenciesHistory(source_values)

    epsilon_values = perform_epsilon_optimization(base_vertices=all_nodes,
                                                  consistencies_history=consistencies_history,
                                                  use_historical=use_historical,
                                                  temporal_window=temporal_window,
                                                  temporal_offset=temporal_offset,
                                                  values_history=values_history)
