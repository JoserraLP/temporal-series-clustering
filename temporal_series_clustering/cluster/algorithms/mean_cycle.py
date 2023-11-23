import copy

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from temporal_series_clustering.cluster.utils import find_key_of_item


def get_cycle_consistencies(cycle: list, consistencies: dict) -> list:
    """
    Get the consistencies values for each item of the cycle

    :param cycle: nodes cycle
    :type cycle: list
    :param consistencies: dictionary with the consistencies
    :type consistencies: dict
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
            if key1 in consistencies:
                values.append(consistencies[key1])
            elif key2 in consistencies:
                values.append(consistencies[key2])
    return values


def get_cycles_info_from_graph(G: nx.MultiDiGraph, consistencies: dict) -> list:
    """
    Retrieve the cycles from a graph and sort them based on their mean consistency

    :param G: graph
    :type G: nx.MultiDiGraph
    :param consistencies: dictionary with the consistencies between nodes
    :type consistencies: dict
    :return: mean-based sorted graph cycles list
    """
    # Get all the possible cycles on the graph
    cycles = get_valid_cycles(G)

    # Calculate the mean value of the cycle consistency
    mean_consistency_cycles = []
    for cycle in cycles:
        # Retrieve the cycle consistencies per each cycle
        cycle_consistencies = get_cycle_consistencies(cycle, consistencies)
        # Calculate the mean value of the consistency. If not valid set infinite as we search the minimum
        mean_consistency = sum(cycle_consistencies) / len(cycle_consistencies) if cycle_consistencies else np.inf
        # Append the mean consistency
        mean_consistency_cycles.append(mean_consistency)

    # Order the cycles by their mean

    # Create a list of tuples where each tuple is (mean, cycle)
    cycles_tuples = list(zip(mean_consistency_cycles, cycles))

    # Sort the list of tuples
    return sorted(cycles_tuples)


def calculate_mean_consistency_from_node(node, possible_clusters, consistencies):
    mean_consistencies = []

    for cluster in possible_clusters:
        mean_value = []
        for cluster_node in cluster:
            if cluster_node != node:
                mean_value.append(consistencies[f"{cluster_node}_{node}"] if f"{cluster_node}_{node}" in consistencies
                                  else consistencies[f"{node}_{cluster_node}"])

        mean_consistencies.append(sum(mean_value) / len(mean_value))

    return mean_consistencies


def get_conflicting_clusters(cycles):
    # Create a set to store elements that appear in more than one list
    conflicting_nodes = set()

    # Create a set to store elements that have been seen
    seen_nodes = set()

    # Retrieve conflicting nodes
    # Iterate over each list and each node in the list
    for _, cycle in cycles:
        for node in cycle:
            # If the node has been seen before, add it to the conflicting nodes set
            if node in seen_nodes:
                conflicting_nodes.add(node)
            # Otherwise, add it to the seen nodes set
            else:
                seen_nodes.add(node)

    # Initialize an empty dictionary to store the results
    conflicting_clusters = {}
    # Iterate over each cycle in the list of cycles
    for _, cycle in cycles:
        # Iterate over the nodes of a cycle
        for node in cycle:
            # If node is conflicting
            if node in conflicting_nodes:
                # If not stored previously
                if node not in conflicting_clusters:
                    # Create a list
                    conflicting_clusters[node] = []
                # Append to list
                conflicting_clusters[node].append(cycle)

    return conflicting_clusters


def get_clusters_from_cycles(cycles: list[tuple], base_vertices: list, G: nx.MultiDiGraph,
                             consistencies: dict, instant: int, historical_clusters: dict = None) -> dict:
    """
    Get the clusters from the cycles of the graph

    :param cycles: all cycles along with its mean consistency value
    :type cycles: list[tuple]
    :param base_vertices: base vertices of the full graph
    :type base_vertices: list
    :param G: graph
    :type G: nx.MultiDiGraph
    :param historical_clusters: clusters for the previous instant. If None, do not enabled temporal clustering.
    Default to None.
    :type historical_clusters: dict
    :return: dict with the clusters where there is a representative and the nodes belonging to that cluster
    """

    # Get conflicting clusters
    conflicting_clusters = get_conflicting_clusters(cycles)

    historical_info_used = []

    # Check if there are conflicting nodes
    if conflicting_clusters:

        clusters = {}
        # Iterate over all conflicting clusters
        for conflicting_node, possible_conflicting_clusters in conflicting_clusters.items():
            check_distance = False
            # Check if the historical info has the conflicting node on the values (not keys)
            if historical_clusters and conflicting_node in \
                    [node for cluster in historical_clusters.values() for node in cluster]:
                # Retrieve the cluster it belongs to
                previous_cluster = [v for k, v in historical_clusters.items() if conflicting_node in v]

                # If there is a cluster
                if previous_cluster:
                    # If there is only one item on the cluster, it should be better to add to a new one, the closest
                    # one, so enable the flag for checking distance
                    if len(previous_cluster[0]) == 1:
                        check_distance = True
                    else:
                        # Store the clusters as the previous value
                        clusters[conflicting_node] = previous_cluster[0]
                        # Append the instant when the historical information have been used
                        historical_info_used.append(instant)
            else:
                check_distance = True

            if check_distance:
                # Check to which cluster is closest, based on mean of consistency
                mean_consistencies_node = calculate_mean_consistency_from_node(conflicting_node,
                                                                               possible_conflicting_clusters,
                                                                               consistencies)

                # Get the index with lowest mean
                argmin = np.argmin(mean_consistencies_node)

                clusters[conflicting_node] = possible_conflicting_clusters[argmin]

        # Remove those nodes that have been stored previously and do something similar as down
        remaining_nodes = set(base_vertices) - set([node for k, v in clusters.items() for node in v])

        # Check if there is some cycle with the remaining nodes
        # Find the value and list where the nodes appear
        remaining_cycles = {node: [item[1] for item in cycles if node in item[1]] for node in remaining_nodes}
        # If there are cycles store like there are
        for node, cycle in remaining_cycles.items():
            if cycle:
                # Remove the conflicting nodes if exists
                cycle = [item for item in cycle[0] if item not in list(conflicting_clusters.keys())]
                # Check if the cycle does not exist
                if cycle not in list(clusters.values()):
                    # Store the cluster
                    clusters[node] = cycle
            else:
                # Single cluster
                clusters[node] = [node]

    else:
        # First initialize the clusters that are alone, not in the graph (which is filtered)
        clusters = {item: [item] for item in set(base_vertices) - set(G.nodes)}

        # Iterate over the cycles
        for mean_cycle, cycle in cycles:
            # First check if the cycle is superset of the clusters
            # Parse cycle to a set
            set_cycle = set(cycle)
            # Define a key for superset if found
            superset_key = ''
            # Iterate over the clusters to check if the cycle is a superset
            for key, cluster_nodes in clusters.items():
                # If the cycle set is a superset of the cluster nodes
                if set_cycle.issuperset(set(cluster_nodes)):
                    # Set the superset_key and stop iterating
                    superset_key = key
                    break

            if superset_key:
                # If it is a superset of any cluster, replace the previous value with the new cluster
                clusters[superset_key] = cycle
            else:
                # Otherwise, get the nodes that are not represented yet in the clusters

                # Flatten the list of values in the dictionary
                existing_clusters_nodes = [item for sublist in clusters.values() for item in sublist]

                # Filter the input cycle, so we have those nodes that are not previously stored in clusters
                non_existing_nodes_cycle = [node for node in cycle if node not in existing_clusters_nodes]

                # If there exist a cycle
                if non_existing_nodes_cycle:
                    # Add the cycle to the clusters
                    clusters[non_existing_nodes_cycle[0]] = non_existing_nodes_cycle

    return clusters, historical_info_used


def insert_intra_cluster_mean(clusters_info: dict, clusters_nodes: dict, consistencies: dict):
    """
    Store the intra consistency mean per each cluster and the overall value for a given instant. Intra is related
    to nodes from the same cluster

    :param clusters_info: input and output clusters information where the intra cluster mean will be stored
    :type clusters_info: dict
    :param clusters_nodes: dict with the clusters nodes
    :type clusters_nodes: dict
    :param consistencies: consistency dictionary
    :type consistencies: dict
    :return: None as the clusters_info is input/output dict
    """
    # Iterate over the clusters nodes to get the mean consistency for a given instant
    for representative, cluster in clusters_nodes.items():
        # Get consistencies of the cluster
        cluster_consistencies = get_cycle_consistencies(cluster, consistencies)
        # Calculate the mean value of the consistency. If not valid set 0.0
        intra_mean_consistency_cluster = sum(cluster_consistencies) / len(cluster_consistencies) \
            if cluster_consistencies else 0.0

        # Append data into the specific cluster on clusters_info
        clusters_info['value'][representative] = {
            'nodes': cluster,
            'intra_mean': intra_mean_consistency_cluster
        }

    # Retrieve the mean values for each clusters
    intra_mean_clusters_values = [v['intra_mean'] for k, v in clusters_info['value'].items() if v['intra_mean']]

    # Store the overall intra mean of a given instant
    clusters_info['intra_mean'] = sum(intra_mean_clusters_values) / len(intra_mean_clusters_values) \
        if len(intra_mean_clusters_values) > 0 else 0.0


def insert_inter_cluster_mean(clusters_info: dict, clusters_nodes: dict, consistencies: dict, G: nx.MultiDiGraph):
    """
    Store the inter consistency mean per each cluster and the overall value for a given instant. Inter is related to
    all connection from each cluster item compared to all the other items from the rest of clusters without considering
    its own cluster nodes

    :param clusters_info: input and output clusters information where the inter cluster mean will be stored
    :type clusters_info: dict
    :param clusters_nodes: dict with the clusters nodes
    :type clusters_nodes: dict
    :param consistencies: consistency dictionary
    :type consistencies: dict
    :param G: non-filtered graph
    :type G: nx.MultiDiGraph
    :return: None as the clusters_info is input/output dict
    """
    # Initialize a variable to sum up all the consistencies
    inter_consistencies_values = []

    # Initialize a list for those connections checked
    connections_checked = []
    # Iterate over all the nodes with the full graph, not the filtered
    for node in G.nodes:
        # Create a list for those nodes belonging to the same cluster
        nodes_same_cluster = []

        # Retrieve the cluster it belongs to
        for k, cluster in clusters_nodes.items():
            if node in cluster:
                nodes_same_cluster = copy.deepcopy(cluster)
                if node in nodes_same_cluster:
                    nodes_same_cluster.remove(node)

        # Retrieve all those nodes that are not in the same cluster
        inter_nodes = [item for item in G.nodes if item != node and item not in nodes_same_cluster]
        # Iterate over the nodes
        for inter_node in inter_nodes:
            # Get keys for both possible directions
            key1 = node + '_' + inter_node
            key2 = inter_node + '_' + node
            # Store the value that exists and if its not stored previously
            if key1 in consistencies and key1 not in connections_checked:
                # Append consistencies value
                inter_consistencies_values.append(consistencies[key1])
                # Append the connection as it has been checked
                connections_checked.append(key1)
            elif key2 in consistencies and key2 not in connections_checked:
                # Append consistencies value
                inter_consistencies_values.append(consistencies[key2])
                # Append the connection as it has been checked
                connections_checked.append(key2)


    """
    # Calculate the overall inter mean of a given instant
    clusters_info['inter_mean'] = sum(inter_consistencies_values) / len(inter_consistencies_values) \
        if len(inter_consistencies_values) > 0 else 0.0
    """
    # Instead of mean, get the minimum
    clusters_info['inter_mean'] = min(inter_consistencies_values) if len(inter_consistencies_values) > 1 else 0.0


def mean_cycle_clustering(G: nx.MultiDiGraph, **kwargs) -> dict:
    """
    Perform the mean cycle clustering algorithm

    :param G: sheaf model simplified for a given instant
    :type G: nx.MultiDiGraph
    :param kwargs: additional parameters such as 'base_vertices', 'epsilon', 'consistency' and
        'prev_historical_clusters'

    :return: clusters for a given instant
    :rtype: dict
    """
    # Get base_vertices, epsilon and the consistency dict
    base_vertices = kwargs['base_vertices']
    epsilon = kwargs['epsilon']
    consistencies = kwargs['consistency']
    historical_clusters = kwargs['historical_clusters'] if 'historical_clusters' in kwargs else None
    instant = kwargs['instant']
    historical_info_used = []

    # STEP 1: CREATE SIMPLIFIED GRAPH
    # From the graph, create a simplified version where there are only store those nodes with lower value than epsilon
    filtered_G = filter_edges_by_consistency(G, epsilon)

    # Add new edges for making the graph connected on both directions
    filtered_G = add_bidirectional_edges(filtered_G)

    # STEP 2: CYCLES
    # Get the cycles of the graph with its mean consistency value
    sorted_cycles = get_cycles_info_from_graph(filtered_G, consistencies)

    # STEP 3: CLUSTERS
    # Get all the clusters from the cycles
    clusters, historical_info_value = get_clusters_from_cycles(cycles=sorted_cycles, base_vertices=base_vertices,
                                                               G=filtered_G,
                                                               consistencies=consistencies,
                                                               historical_clusters=historical_clusters,
                                                               instant=instant)

    historical_info_used.extend(historical_info_value)

    # STEP 4: CLUSTERS ADDITIONAL INFO
    # Define the clusters info dict
    clusters_info = {'value': {}}
    # Insert intra cluster mean into the clusters_info
    insert_intra_cluster_mean(clusters_info=clusters_info, clusters_nodes=clusters, consistencies=consistencies)
    # Insert inter cluster mean into the clusters_info, using the full graph, not the filtered
    insert_inter_cluster_mean(clusters_info=clusters_info, clusters_nodes=clusters, consistencies=consistencies, G=G)
    # Insert silhouette score
    # insert_silhouette_score(clusters_info=clusters_info, clusters_nodes=clusters, consistencies=consistencies)

    return clusters_info, historical_info_used


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
            min_length_cycles = find_min_length_cycles(G, node)
            # Get only those that have the node
            all_min_length_cycles[node] = [cycle for cycle in min_length_cycles if node in cycle]

    # Parse to list
    all_combined_cycles = []
    for k in list(all_min_length_cycles.keys()):
        # Store the combined cycles
        all_combined_cycles.append(find_combined_cycle(all_min_length_cycles[k]))

    return all_combined_cycles
