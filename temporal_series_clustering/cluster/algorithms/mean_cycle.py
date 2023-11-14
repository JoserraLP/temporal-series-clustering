import networkx as nx
import numpy as np
import pandas as pd

from temporal_series_clustering.cluster.utils import find_key_of_item


def mean_cycle_clustering(G: nx.MultiDiGraph, **kwargs) -> dict:
    """
    Perform the mean cycle clustering algorithm

    :param G: sheaf model simplified for a given instant
    :type G: nx.MultiDiGraph
    :param kwargs: additional parameters such as 'base_vertices', 'epsilon', and 'consistency'

    :return: clusters for a given instant
    :rtype: dict
    """
    # Get base_vertices, epsilon and the consistency dict
    base_vertices = kwargs['base_vertices']
    epsilon = kwargs['epsilon']
    consistency_dict = kwargs['consistency']

    # STEP 1: CREATE SIMPLIFIED GRAPH
    # From the graph, create a simplified version where there are only store those nodes with lower value than epsilon
    filtered_G = filter_edges_by_consistency(G, epsilon)

    # Add new edges for making the graph connected on both directions
    filtered_G = add_bidirectional_edges(filtered_G)

    # STEP 2: INITIALIZE CLUSTERS
    # Get those nodes that are not in the filtered graph as they belong to separate clusters
    clusters = {item: [item] for item in set(base_vertices) - set(filtered_G.nodes)}

    # STEP 3: CYCLES
    # Get all the possible cycles on the graph
    cycles = get_valid_cycles(filtered_G)

    # Calculate the mean value of the cycle consistency
    mean_consistency_cycles = []
    for cycle in cycles:
        # Retrieve the cycle consistencies
        cycle_consistencies = get_consistencies_cycle(cycle, consistency_dict)
        # Calculate the mean value of the consistency. If not valid set infinite as we search the minimum
        mean_consistency = sum(cycle_consistencies) / len(cycle_consistencies) if cycle_consistencies else np.inf
        # Append the mean consistency
        mean_consistency_cycles.append(mean_consistency)

    # Get the cycle index with the lowest mean consistency
    min_consistency_cluster_idx = pd.Series(mean_consistency_cycles).idxmin()

    # Append the cycle to the cluster, first item as represent
    clusters[cycles[min_consistency_cluster_idx][0]] = cycles[min_consistency_cluster_idx]

    # Remove the cycle from the list of cycles as it has been checked
    cycles.pop(min_consistency_cluster_idx)

    # Iterate over the rest of the cycles
    for cycle in cycles:

        # First check if the cycle is superset of the clusters
        # Parse cycle to a set
        set_cycle = set(cycle)
        # Define a key for superset if found
        superset_key = ''
        # Iterate over the clusters
        for key, value in clusters.items():
            # If the cycle set is a superset of the current value
            if set_cycle.issuperset(set(value)):
                # Set the superset_key and stop iterating
                superset_key = key
                break

        if superset_key:
            # If it is a superset of any cluster, replace the previous value with the new cluster
            clusters[superset_key] = cycle
        else:
            # Otherwise, get the nodes that are not represented yet in the clusters

            # Flatten the list of values in the dictionary
            all_clusters_values = [item for sublist in clusters.values() for item in sublist]

            # Filter the input cycle so we have those values that are not previously stored in clusters
            updated_cycle = [item for item in cycle if item not in all_clusters_values]

            # If there exist a cycle
            if updated_cycle:
                # Add the cycle to the clusters
                clusters[updated_cycle[0]] = updated_cycle

    # Define the clusters info dict
    clusters_info = {'value': {}, 'mean': None}

    # Iterate over the clusters to get the mean consistency for a given instant
    for k, cluster in clusters.items():
        # Get consistencies of the cluster
        cluster_consistencies = get_consistencies_cycle(cluster, consistency_dict)
        # Calculate the mean value of the consistency. If not valid set infinite as we search the minimum
        mean_consistency_cluster = sum(cluster_consistencies) / len(cluster_consistencies) if cluster_consistencies \
            else None
        clusters_info['value'][k] = {
            'nodes': cluster,
            'mean': mean_consistency_cluster
        }

    # Retrieve the mean values for each clusters
    mean_clusters_values = [v['mean'] for k, v in clusters_info['value'].items() if v['mean']]

    # Calculate the overall mean of a given instant
    clusters_info['mean'] = sum(mean_clusters_values) / len(mean_clusters_values) if len(mean_clusters_values) > 0 \
        else None

    return clusters_info


def get_consistencies_cycle(cycle: list, consistencies_dict: dict) -> list:
    """
    Get the consistencies values for each item of the cycle
    :param cycle: nodes cycle
    :type cycle: list
    :param consistencies_dict: dictionary with the consistencies
    :type consistencies_dict: dict
    :return: list with consistencies of the cycle nodes
    :rtype: list
    """
    # Define the values list
    values = []
    # Iterate over all elements of the cycle and its combinations
    for i in range(len(cycle)):
        for j in range(i + 1, len(cycle)):
            # Get keys for both possible directions
            key1 = cycle[i] + '_' + cycle[j]
            key2 = cycle[j] + '_' + cycle[i]
            # Store the value that exists
            if key1 in consistencies_dict:
                values.append(consistencies_dict[key1])
            elif key2 in consistencies_dict:
                values.append(consistencies_dict[key2])
    return values


def filter_edges_by_consistency(G: nx.MultiDiGraph, epsilon: float) -> nx.MultiDiGraph:
    """
    Creates a new graph where there are only store the edges with a consistency value below epsilon

    :param G: input graph
    :type G: nx.MultiDiGraph
    :param epsilon: threshold value
    :type epsilon: float
    :return: graph with filtered edges
    :rtype: nx.MultiDiGraph
    """
    # Create a new graph
    H = nx.MultiDiGraph()

    # Iterate over the edges in the original graph with the data
    for u, v, data in G.edges(data=True):
        # If the 'consistency' value is lower than the threshold, add the edge to the new graph
        if data.get('consistency', float('inf')) < epsilon:
            H.add_edge(u, v, **data)

    return H


def add_bidirectional_edges(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Add bidirectional edges on a given single-directed graph

    :param G: input graph
    :type G: nx.MultiDiGraph
    :return: updated graph
    :rtype: nx.MultiDiGraph
    """
    # Iterate over the edges in the original graph with the data
    for edge in list(G.edges(data=True)):
        G.add_edge(edge[1], edge[0], **edge[2])

    return G


def find_min_length_cycles(G: nx.MultiDiGraph, source: str) -> list:
    """
    Find the minimum length cycles higher than 2 of length (avoiding leaves)

    :param G: input graph
    :type G: nx.MultiDiGraph
    :param source: source node id
    :type source: str
    :return: list with minimum length cycles
    :rtype: list
    """
    # Initialize a list to store all cycles
    all_cycles = []

    # Use Depth First Search  algorithm

    # Recursive DFS function
    def dfs(path):
        # Get the last node in the path
        node = path[-1]

        # Get the neighbors of the node
        neighbors = G.neighbors(node)

        # Recursively call dfs for each neighbor that is not already in the path
        for neighbor in neighbors:
            if neighbor not in path:
                dfs(path + [neighbor])
            elif neighbor == source and len(path) > 2:
                # If we've reached the source node and the path contains more than one node,
                # add the current path to all_cycles
                all_cycles.append(path + [neighbor])

    # Call the DFS function with the source node as the starting path
    dfs([source])

    # Check if there are cycles
    if all_cycles:
        # Find the cycles with the minimum length
        min_length = min(len(cycle) for cycle in all_cycles)
        # Remove the last item of the cycle as it is the same as the first one
        min_length_cycles = [cycle[:-1] for cycle in all_cycles if len(cycle) == min_length]
    else:
        # If there are no cycles, check the leaves and append them into the cycles
        min_length_cycles = [(node, list(G.neighbors(node))[0]) for node in G.nodes() if G.out_degree(node) == 1]

    return min_length_cycles


def find_combined_cycle(cycles: list) -> list:
    """
    Find the cycle that is combination of the input cycles if exists

    :param cycles: cycles to check
    :type cycles: list
    :return: list of combined cycles
    :rtype: lsit
    """
    # Initialize an empty set to store the unique nodes
    unique_nodes = set()

    # Iterate over all the cycles
    for cycle in cycles:
        # Add the nodes of the current cycle to the set of unique nodes
        unique_nodes.update(cycle)

    # Convert the set of unique nodes back to a list
    combined_cycle = list(unique_nodes)

    return combined_cycle


def get_valid_cycles(G: nx.MultiDiGraph) -> list:
    """
    Get valid cycles (those with minimum length and its combinations)

    :param G: input graph
    :type G: nx.MultiDiGraph
    :return: list with the cycles
    :rtype: list
    """
    # Initialize the variable
    all_min_length_cycles = {}
    # Get all minimum length cycles for nodes on simplified graph
    for node in G.nodes():
        # Check if node not in some of the previous cycles
        keys = find_key_of_item(node, all_min_length_cycles)
        # If it is not stored previously
        if keys is None:
            # Retrieve the minimum length cycles
            all_min_length_cycles[node] = find_min_length_cycles(G, node)

    # Parse to list
    all_combined_cycles = []
    for k in list(all_min_length_cycles.keys()):
        # Store the combined cycles
        all_combined_cycles.append(find_combined_cycle(all_min_length_cycles[k]))

    return all_combined_cycles
