import copy

from temporal_series_clustering.cluster.algorithms import epsilon_paths
from temporal_series_clustering.cluster.algorithms.dbscan import dbscan_cluster
from temporal_series_clustering.cluster.algorithms.min_path import min_path_clustering


def temporal_clustering(temporal_sheaf_models, temporal_consistency: dict, base_vertices: list,
                        epsilon: float, algorithm) -> dict:
    # Define the clusters
    clusters = {}

    algorithm_name = ''
    if algorithm == min_path_clustering:
        algorithm_name = 'min_path'
    elif algorithm == dbscan_cluster:
        algorithm_name = 'dbscan'
    elif algorithm == epsilon_paths:
        algorithm_name = 'epsilon_paths'

    # Copy the temporal_consistency dict as we are going to remove some values
    temporal_consistency_copy = copy.deepcopy(temporal_consistency)

    # Iterate over the instants
    for instant, instant_consistency_value in temporal_consistency_copy.items():
        # Remove the output instant consistency node value '@'
        instant_consistency_value.pop('@')

        # Retrieve graph
        G = temporal_sheaf_models[instant]

        # Store the clusters
        clusters[instant] = algorithm(G, **{'base_vertices': base_vertices, 'epsilon': epsilon,
                                            'consistency': instant_consistency_value})

        # Show a graph with the nodes
        # show_graph(G, clusters=clusters[instant], instant=instant, method=algorithm_name)

    return clusters
