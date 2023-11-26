import networkx as nx
import numpy as np

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


def find_combined_cycle(cycles: list) -> list:
    """
    Find the cycle that is combination of the input cycles if exists

    :param cycles: cycles to check
    :type cycles: list
    :return: list of combined cycles
    :rtype: list
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


def filter_edges_by_epsilon(G: nx.MultiDiGraph, epsilon: int) -> nx.MultiDiGraph:
    """
    Creates a new graph where there are only store the edges with a consistency value below epsilon and add the
    bidirectional edges

    :param G: input graph
    :type G: nx.MultiDiGraph
    :param epsilon: threshold
    :type epsilon: int
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
            # Also add the bidirectional edge
            H.add_edge(v, u, **data)

    return H


def get_cycles_info_from_graph(G: nx.MultiDiGraph, instant_consistencies: dict) -> list:
    """
    Retrieve the cycles from a graph and sort them based on their mean consistency

    :param G: graph
    :type G: nx.MultiDiGraph
    :param instant_consistencies: current instant consistencies
    :type instant_consistencies: dict
    :return: mean-based sorted graph cycles list
    """
    # Get all the possible cycles on the graph
    cycles = get_valid_cycles(G)

    # Calculate the mean value of the cycle consistency
    mean_consistency_cycles = []
    for cycle in cycles:
        # Retrieve the cycle consistencies per each cycle
        cycle_consistencies = get_cycle_consistencies(cycle, instant_consistencies)
        # Calculate the mean value of the consistency. If not valid set infinite as we search the minimum
        mean_consistency = sum(cycle_consistencies) / len(cycle_consistencies) if cycle_consistencies else np.inf
        # Append the mean consistency
        mean_consistency_cycles.append(mean_consistency)

    # Order the cycles by their mean

    # Create a list of tuples where each tuple is (mean, cycle)
    cycles_tuples = list(zip(mean_consistency_cycles, cycles))

    # Sort the list of tuples
    return sorted(cycles_tuples)


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


def calculate_mean_consistency_from_node(node, possible_clusters, consistencies):
    mean_consistencies = []

    for cluster in possible_clusters:
        mean_value = []
        for cluster_node in cluster:
            if cluster_node != node:
                mean_value.append(consistencies[f"{cluster_node}_{node}"] if f"{cluster_node}_{node}" in consistencies
                                  else consistencies[f"{node}_{cluster_node}"])

        mean_consistencies.append(sum(mean_value) / len(mean_value) if len(mean_value) > 0 else 0.0)

    return mean_consistencies


def get_subset_conflicting_clusters(conflicting_clusters):
    # Check if the separate conflicting clusters have the same values, so it is possible to group them
    # Remove the key from each list in the dictionary
    clusters = {key: [item for item in value if item != key] for key, value in conflicting_clusters.items()}

    # Find keys that have the same subset
    result = {}
    for key, value in clusters.items():
        value = tuple(sorted(value))  # Convert list to tuple so it can be used as a dictionary key
        if value not in result:
            result[value] = []
        result[value].append(key)

    # Combine the keys and their corresponding lists
    combined_result = {keys[0]: sorted(list(value) + keys) for value, keys in result.items()}

    # If there is not combination, return the input
    if not combined_result:
        combined_result = conflicting_clusters

    return combined_result


def get_clusters_from_cycles(base_vertices: list, graph_nodes: list, cycles: list[tuple], instant: int,
                             instant_consistencies: dict, previous_clusters_nodes: dict = None) -> dict:
    """
    Get the clusters from the cycles of the graph

    :param base_vertices: list with base vertices
    :type base_vertices: list
    :param graph_nodes: graph nodes
    :type graph_nodes: list
    :param cycles: all cycles along with its mean consistency value
    :type cycles: list[tuple]
    :param instant: instant for clustering
    :type instant: int
    :param instant_consistencies: instant consistencies dict
    :type instant_consistencies: dict
    :param previous_clusters_nodes: clusters for the previous instant. If None, do not enabled temporal clustering.
    Default to None.
    :type previous_clusters_nodes: dict
    :return: dict with the clusters where there is a representative and the nodes belonging to that cluster
    """
    # Get conflicting clusters
    conflicting_clusters = get_conflicting_clusters(cycles)

    historical_info_used = []

    # Check if there are conflicting nodes
    if conflicting_clusters:
        print("-" * 10)
        print(f"INSTANT {instant}")
        print(f"Conflicting nodes are: {conflicting_clusters.keys()}")
        clusters = {}
        # Iterate over all conflicting clusters
        for conflicting_node, possible_conflicting_clusters in conflicting_clusters.items():
            check_distance = False
            # Check if the historical info has the conflicting node on the values (not keys)
            if previous_clusters_nodes and conflicting_node in \
                    [node for cluster in previous_clusters_nodes.values() for node in cluster]:

                print(f"Conflicting node {conflicting_node} has clusters {possible_conflicting_clusters} ")
                # Retrieve the cluster it belongs to
                previous_cluster = [v for k, v in previous_clusters_nodes.items() if conflicting_node in v]

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
                        print(f"Using historical info on {instant} with cluster {previous_cluster[0]}")
            else:
                check_distance = True

            if check_distance:
                # Get all those remaining nodes that are conflicting as they will be checked afterwards
                remaining_conflicting_nodes = [node for node in conflicting_clusters.keys() if node != conflicting_node]
                # Remove them from the conflicting clusters
                possible_conflicting_clusters = [[item for item in sublist if item not in remaining_conflicting_nodes]
                                                 for sublist in possible_conflicting_clusters]

                # Check to which cluster is closest, based on mean of consistency
                mean_consistencies_node = calculate_mean_consistency_from_node(conflicting_node,
                                                                               possible_conflicting_clusters,
                                                                               consistencies=instant_consistencies)

                # Get the index with lowest mean
                argmin = np.argmin(mean_consistencies_node)

                clusters[conflicting_node] = possible_conflicting_clusters[argmin]

        # Gather subsets into bigger sets
        clusters = get_subset_conflicting_clusters(clusters)

        print(f"Clusters based on conflicting nodes are: {clusters}")

        # Remove those nodes that have been stored previously and do something similar as down
        remaining_nodes = set(base_vertices) - set([node for k, v in clusters.items() for node in v])

        print(f"Remaining nodes are: {remaining_nodes}")

        # Check if there is some cycle with the remaining nodes
        # Find the value and list where the nodes appear
        remaining_cycles = {node: [item[1] for item in cycles if node in item[1]] for node in remaining_nodes}

        # If there are cycles store like there are
        for node, cycle in remaining_cycles.items():
            print(f"Node {node}")
            if cycle:
                print(f"Processing cycle {cycle}")
                # Remove the conflicting nodes if exists
                cycle = [item for item in cycle[0] if item not in list(conflicting_clusters.keys())]
                print(f"Cycle - {cycle}")
                # Check if the cycle does not exist
                if cycle not in list(clusters.values()):
                    # Store the cluster
                    clusters[node] = cycle
            else:
                # Single cluster
                clusters[node] = [node]

        print(f"Final clusters are: {clusters}")

    else:
        # First initialize the clusters that are alone, not in the graph (which is filtered)
        clusters = {item: [item] for item in set(base_vertices) - set(graph_nodes)}

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
