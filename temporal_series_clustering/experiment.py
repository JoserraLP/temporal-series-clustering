import json
import string
import time

import numpy as np

from temporal_series_clustering.cluster.algorithms.tcbc import TCBC
from temporal_series_clustering.cluster.utils import store_clusters_json
from temporal_series_clustering.patterns.generators import predictor_city, \
    create_simulation_specific_weekday, add_instant_variation, add_offset, smooth_edgy_values, combine_patterns
from temporal_series_clustering.sheaf.sheaf_model import create_sheaf_model
from temporal_series_clustering.sheaf.utils import propagate_sheaf_values
from temporal_series_clustering.static.constants import ITEMS_PER_DAY
from temporal_series_clustering.storage.clusters_history import ClustersHistory
from temporal_series_clustering.storage.consistencies import ConsistenciesHistory
from temporal_series_clustering.storage.epsilons import EpsilonValues
from temporal_series_clustering.storage.simplified_graphs import SimplifiedGraphsHistory


def generate_patterns(total_num=50):
    # Get the simulations of each predictor
    a_simulation = create_simulation_specific_weekday(predictor_city, place_id='a', weekday="weekday")
    b_simulation = create_simulation_specific_weekday(predictor_city, place_id='b', weekday="weekday")
    c_simulation = create_simulation_specific_weekday(predictor_city, place_id='c', weekday="weekday")
    d_simulation = create_simulation_specific_weekday(predictor_city, place_id='d', weekday="weekday")
    e_simulation = create_simulation_specific_weekday(predictor_city, place_id='e', weekday="weekday")
    f_simulation = create_simulation_specific_weekday(predictor_city, place_id='f', weekday="weekday")

    predictors_output = [a_simulation, b_simulation, c_simulation, d_simulation, e_simulation, f_simulation]

    # 6 patterns
    # Get patterns with noise
    instant_noise = 0.05
    a_noise = add_instant_variation(a_simulation, noise_level=instant_noise)
    b_noise = add_instant_variation(b_simulation, noise_level=instant_noise)
    c_noise = add_instant_variation(c_simulation, noise_level=instant_noise)
    d_noise = add_instant_variation(d_simulation, noise_level=instant_noise)
    e_noise = add_instant_variation(e_simulation, noise_level=instant_noise)
    f_noise = add_instant_variation(f_simulation, noise_level=instant_noise)

    predictors_output.extend([a_noise, b_noise, c_noise, d_noise, e_noise, f_noise])

    # 12 patterns
    # Less noise
    instant_noise = 0.01
    a_noise = add_instant_variation(a_simulation, noise_level=instant_noise)
    b_noise = add_instant_variation(b_simulation, noise_level=instant_noise)
    c_noise = add_instant_variation(c_simulation, noise_level=instant_noise)
    d_noise = add_instant_variation(d_simulation, noise_level=instant_noise)
    e_noise = add_instant_variation(e_simulation, noise_level=instant_noise)
    f_noise = add_instant_variation(f_simulation, noise_level=instant_noise)

    predictors_output.extend([a_noise, b_noise, c_noise, d_noise, e_noise, f_noise])

    # 18 patterns

    # Add offset for original and noise patterns

    np.random.seed(1)
    offset = np.random.randint(-2, 2)
    a_offset = add_offset(a_simulation, offset=offset)
    offset = np.random.randint(-2, 2)
    b_offset = add_offset(b_simulation, offset=offset)
    offset = np.random.randint(-2, 2)
    c_offset = add_offset(c_simulation, offset=offset)
    offset = np.random.randint(-2, 2)
    d_offset = add_offset(d_simulation, offset=offset)
    offset = np.random.randint(-2, 2)
    e_offset = add_offset(e_simulation, offset=offset)
    offset = np.random.randint(-2, 2)
    f_offset = add_offset(f_simulation, offset=offset)

    predictors_output.extend([a_offset, b_offset, c_offset, d_offset, e_offset, f_offset])

    # 24 patterns

    # Combine remaining patterns
    pattern_combinations = []
    for i in range(total_num - len(predictors_output)):
        # one more than the length as we want one more item due to range behavior
        pattern_combinations.append(smooth_edgy_values(combine_patterns(predictors_output, seed=i)))

    predictors_output.extend(pattern_combinations)

    return predictors_output


def perform_epsilon_optimization(consistencies_history: ConsistenciesHistory, base_vertices):
    # Define epsilon values object
    epsilon_values = EpsilonValues()

    # Create the simplified graphs
    simplified_graphs = SimplifiedGraphsHistory(num_base_vertices=len(base_vertices))

    simplified_graphs.create_simplified_graphs(consistencies_history=consistencies_history)

    total_times = {}

    # Perform clustering per each possible epsilon (represented as int)
    for epsilon in np.arange(0.05, 0.20, 0.05):

        start_time = time.time()
        tcbc = TCBC(consistencies_history=consistencies_history,
                    clusters_history=ClustersHistory(),
                    simplified_graphs_history=simplified_graphs,
                    base_vertices=base_vertices,
                    epsilon=epsilon)

        historical_info_used = tcbc.perform_all_instants_clustering()

        epsilon_values.insert_all_info_on_epsilon(epsilon=epsilon, historical_info_used=historical_info_used,
                                                  clusters_history=tcbc.clusters_history)

        total_times[epsilon] = time.time() - start_time
        print(f"On epsilon {epsilon}, lasted time {time.time() - start_time}")

        # Store epsilon file
        store_clusters_json(epsilon_values.info, f'../results/experiment_50_clusters.json')

        with open('../results/experiment_50_clusters_times.json', 'w') as f:
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
    # Define number of patterns
    num_patterns = 50
    # Generate patterns
    patterns = generate_patterns(num_patterns)

    # Get base vertices
    base_vertices = generate_characters(num_patterns)
    # Create sheaf model with base vertices
    sheaf_model = create_sheaf_model(base_vertices)

    # For testing purposes we are going to generate the output for each predictor
    reading_freq = 1

    summary, filtration, mean, values = propagate_sheaf_values(sheaf=sheaf_model,
                                                               predictors_output=patterns,
                                                               reading_freq=reading_freq,
                                                               num_items=ITEMS_PER_DAY,
                                                               base_vertices=base_vertices)

    # Create the different storages
    consistencies_history = ConsistenciesHistory(filtration)

    epsilon_values = perform_epsilon_optimization(base_vertices=base_vertices,
                                                  consistencies_history=consistencies_history)
