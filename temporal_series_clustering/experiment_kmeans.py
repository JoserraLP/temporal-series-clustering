import time

import json

from temporal_series_clustering.cluster.algorithms.kmeans import KMeansClustering
from temporal_series_clustering.cluster.cluster_utils import store_clusters_json
from temporal_series_clustering.experiment_tcbc import generate_patterns, generate_characters
from temporal_series_clustering.storage.clusters_history import ClustersHistory
from temporal_series_clustering.storage.epsilons import EpsilonValues

if __name__ == "__main__":
    # Define number of patterns
    num_patterns = 21
    """
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
    patterns = generate_patterns(weekday="weekday", total_num=num_patterns)

    # Get base vertices
    base_vertices = generate_characters(num_patterns)

    total_times = {}

    # Define epsilon values object
    k_values = EpsilonValues()

    # Perform clustering per each possible epsilon
    for k in range(2, num_patterns):
        start_time = time.time()

        kmeans = KMeansClustering(clusters_history=ClustersHistory(),
                                  base_vertices=base_vertices,
                                  k=k, input_time_series=patterns)

        kmeans.perform_all_instants_clustering()

        k_values.insert_all_info_on_epsilon(epsilon=k, historical_info_used=[],
                                                      clusters_history=kmeans.clusters_history)

        total_times[k] = time.time() - start_time
        print(f"On K {k}, lasted time {time.time() - start_time}")

        # Store epsilon file
        store_clusters_json(k_values.info,
                            f'../results/kmeans_{len(base_vertices)}_sources.json')

        with open(f'../results/kmeans_{len(base_vertices)}_sources_times.json', 'w') as f:
            json.dump(total_times, f)

