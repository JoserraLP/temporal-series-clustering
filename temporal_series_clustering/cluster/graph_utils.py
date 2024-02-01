import copy

import networkx as nx
import numpy as np

from temporal_series_clustering.cluster.utils import all_equal


def get_cliques_from_graph(graph: nx.Graph) -> list:
    """
    Get maximum cliques from a given graph

    :param graph: input undirected graph
    :type graph: nx.Graph
    :return: list with cliques
    :rtype: list
    """
    # Bron–Kerbosch algorithm
    # Find all cliques from the undirected graph
    return list(nx.find_cliques(graph))


def get_clique_mean_value(clique_nodes: list, values: dict):
    """
    Get the clique mean value

    :param clique_nodes: nodes of the clique
    :type clique_nodes: list
    :param values: dictionary with the values
    :type values: dict
    :return: float with mean value of the clique nodes
    :rtype: float
    """
    return np.mean([value for source, value in values.items() if source in clique_nodes])


def get_clique_consistencies(clique_nodes: list, consistencies: dict) -> list:
    """
    Get the consistencies values for each item of the clique

    :param clique_nodes: nodes of the clique
    :type clique_nodes: list
    :param consistencies: dictionary with the consistencies
    :type consistencies: dict
    :return: list with consistencies of the clique nodes
    :rtype: list
    """
    # Define the values list
    values = []
    # Iterate over all elements of the clique and its combinations
    for i in range(len(clique_nodes)):
        for j in range(i + 1, len(clique_nodes)):
            # Get keys for both possible directions
            key1 = clique_nodes[i] + '_' + clique_nodes[j]
            key2 = clique_nodes[j] + '_' + clique_nodes[i]
            # Store the value that exists
            if key1 in consistencies:
                values.append(consistencies[key1])
            elif key2 in consistencies:
                values.append(consistencies[key2])
    return values


def filter_edges_by_epsilon(G: nx.Graph, epsilon: float) -> nx.Graph:
    """
    Creates a new graph where there are only store the edges with a consistency value below epsilon

    :param G: input graph
    :type G: nx.MultiDiGraph
    :param epsilon: threshold
    :type epsilon: float
    :return: graph with filtered edges
    :rtype: nx.Graph
    """
    # Create a new graph
    H = nx.Graph()

    # Iterate over the edges in the original graph with the data
    for u, v, data in G.edges(data=True):
        # If the 'consistency' value is lower than the threshold, add the edge to the new graph
        if data.get('consistency', float('inf')) < epsilon:
            H.add_edge(u, v, **data)
            # Also add the bidirectional edge
            # H.add_edge(v, u, **data)

    return H


def get_cliques_info_from_graph(G: nx.Graph, instant_consistencies: dict) -> list:
    """
    Retrieve the maximum cliques from a graph and sort them based on their mean consistency

    :param G: undirected graph
    :type G: nx.Graph
    :param instant_consistencies: current instant consistencies
    :type instant_consistencies: dict
    :return: mean-based sorted graph cliques list
    """
    # Get all the possible cliques on the graph
    cliques = get_cliques_from_graph(G)

    # Calculate the mean value of the clique consistency
    mean_consistency_cliques = []
    for clique in cliques:
        # Retrieve the clique consistencies per each clique
        clique_consistencies = get_clique_consistencies(clique, instant_consistencies)
        # Calculate the mean value of the consistency. If not valid set infinite as we search the minimum
        mean_consistency = sum(clique_consistencies) / len(clique_consistencies) if clique_consistencies else np.inf
        # Append the mean consistency
        mean_consistency_cliques.append(mean_consistency)

    # Order the cliques by their mean

    # Create a list of tuples where each tuple is (mean, clique)
    cliques_tuples = list(zip(mean_consistency_cliques, cliques))

    # Sort the list of tuples
    return sorted(cliques_tuples)


def get_overlapping_clusters(cliques: list[tuple]) -> dict:
    """
    Get overlapping clusters from all the possible cliques

    :param cliques: cliques with the nodes associated and its mean consistency
    :type cliques: list[tuple]
    :return: dict with overlapping clusters
    :rtype: dict
    """
    # Create a set to store elements that appear in more than one list
    overlapping_nodes = set()

    # Create a set to store elements that have been seen
    seen_nodes = set()

    # Retrieve overlapping nodes
    # Iterate over each list and each node in the list
    for _, clique in cliques:
        for node in clique:
            # If the node has been seen before, add it to the overlapping nodes set
            if node in seen_nodes:
                overlapping_nodes.add(node)
            # Otherwise, add it to the seen nodes set
            else:
                seen_nodes.add(node)

    # Initialize an empty dictionary to store the overlapping clusters
    overlapping_clusters = {}
    # Iterate over each clique in the list of cliques
    for _, clique in cliques:
        # Iterate over the nodes of a clique
        for node in clique:
            # If node is overlapping
            if node in overlapping_nodes:
                # If not stored previously on clusters
                if node not in overlapping_clusters:
                    # Create a list to
                    overlapping_clusters[node] = []
                # Append the clique to the node
                overlapping_clusters[node].append(clique)

    return overlapping_clusters


def recalculate_sort_cliques(cliques: list[tuple], consistencies: dict) -> list[tuple]:
    """
    Recalculate the cliques mean consistency and sort by it

    :param cliques: list with cliques mean and nodes
    :type cliques: list[tuple]
    :param consistencies: consistencies values
    :type consistencies: dict
    :return: sorted cliques with new means as tuples (mean, nodes)
    :rtype: list[tuple]
    """
    # Initialize clique consistencies list
    clique_mean_consistencies = []

    # Iterate over the cliques nodes (left the mean)
    for _, clique_nodes in cliques:
        # Retrieve the consistency for the clique nodes
        clique_nodes_consistencies = get_clique_consistencies(clique_nodes=clique_nodes, consistencies=consistencies)
        # Append the new mean consistency. If there is no value, set to 0
        clique_mean_consistencies.append(sum(clique_nodes_consistencies) / len(clique_nodes_consistencies)
                                         if len(clique_nodes_consistencies) > 0 else 0.00)

    # Create a list of tuples where each tuple is (mean, clique_nodes)
    cliques_tuples = list(zip(clique_mean_consistencies, [clique_nodes for _, clique_nodes in cliques]))

    return sorted(cliques_tuples)


def get_min_length_cliques(cliques: list[tuple]) -> list:
    """
    Retrieve the clique with minimum length (greater than 1) avoiding single nodes clusters

    :param cliques: cliques
    :type cliques: list[tuple]
    :return: clique nodes with minimum length
    :rtype: list
    """
    # Get those cliques with minimum length
    min_length_cliques = [clique_nodes for _, clique_nodes in cliques if len(clique_nodes) > 1]
    # Resulting clique if exists, otherwise empty list
    resulting_clique = min_length_cliques[0] if len(min_length_cliques) > 0 else []
    # Get the resulting clique index from the original list if exists, otherwise -1
    resulting_clique_idx = [clique_nodes for _, clique_nodes in cliques].index(resulting_clique) \
        if len(resulting_clique) > 0 else -1

    return resulting_clique, resulting_clique_idx


def calculate_temporal_binding_factor(overlapping_nodes: dict, previous_clusters: dict):
    """
    Calculate the temporal binding for each possible overlapping nodes

    :param overlapping_nodes: dictionary with each overlapping node and the possible clusters it can belong
    :type overlapping_nodes: dict
    :param previous_clusters: dictionary with clusters (final and overlapping) on previous instants
    :type previous_clusters: dict
    :return: dict with temporal binding probability for each possible cluster on each overlapping node
    """
    # Define the temporal binding probabilities dict
    temporal_binding_probabilities = {}

    # Another approach could be not checking that it is the same cluster but a subset or superset
    # Iterate over the overlapping nodes
    for overlapping_node, possible_clusters in overlapping_nodes.items():
        # Create an empty list for each overlapping node
        temporal_binding_probabilities[overlapping_node] = []

        # Count number of times the node has been overlapping previously (it exists on 'overlapping_clusters' 
        # from previous info)
        overlapping_times = [1 if overlapping_node in info['overlapping_clusters'].keys() else 0
                             for _, info in previous_clusters.items()]

        # Calculate the probability by sum / length
        p_n = sum(overlapping_times) / len(overlapping_times)

        # Set the value of overlapping node to 1 if it does not appear previously. This avoids future errors
        p_n = 1 if p_n == 0 else p_n

        # Define list for both the probability of node n being overlapping and belonging to a given cluster c_t (p_n_ct)
        #  and the probability of a cluster ct to appear (p_ct)
        p_n_ct, p_ct = [], []

        # Iterate over each possible cluster
        for cluster in possible_clusters:
            # Count number of times the node has been on the cluster previously
            node_in_cluster_times = [1 if overlapping_node in info['clusters'] and
                                          set(info['clusters'][overlapping_node]) == set(cluster) else 0
                                     for _, info in previous_clusters.items()]
            # Calculate the probability by sum / length
            p_ct.append(sum(node_in_cluster_times) / len(node_in_cluster_times))

            # Count the number of times the cluster is in previous overlapping clusters and the overlapping node was
            # overlapping previously
            cluster_overlapping_node_times = [1 if overlapping_node in info['overlapping_clusters'] and
                                                   cluster in list(info['overlapping_clusters'][overlapping_node])
                                              else 0 for _, info in previous_clusters.items()]
            # Calculate the probability by sum / length
            p_n_ct.append(sum(cluster_overlapping_node_times) /
                          len(cluster_overlapping_node_times))

        # Calculate the temporal binding probability for each possible cluster on a given overlapping node
        temporal_binding_probabilities[overlapping_node] = [p_n_ct[i] * p_ct[i] / p_n
                                                            for i in range(len(possible_clusters))]

    return temporal_binding_probabilities


def get_clusters_from_cliques(all_nodes: list, graph_nodes: list, cliques: list[tuple], instant: int,
                              instant_consistencies: dict, previous_clusters: dict = None, temporal_offset: int = 0):
    """
    Get clusters from all the possible cliques on a given instant

    :param all_nodes: all nodes existent
    :type all_nodes: list
    :param graph_nodes: filtered graph nodes
    :type graph_nodes: list
    :param cliques: maximum cliques from filtered graph as tuple (mean, cliques_nodes)
    :type cliques: list[tuple]
    :param instant: current instant
    :type instant: int
    :param instant_consistencies: consistencies for the given instant
    :type instant_consistencies: dict
    :param previous_clusters: previous clusters information only enabled if historical info used. Default to None
    :type previous_clusters: dict
    :param temporal_offset: offset to consider current instant on previous full pattern. If 0, do not consider it,
        just use previous instants
    :type temporal_offset: int
    :return: clusters dict, overlapping nodes and historical information used
    """
    # Create a list for storing instants where historical information have been used,
    # a list those nodes that have been checked,
    # a list for storing the final clusters as a list
    historical_info_used, nodes_checked, clusters_list = [], [], []
    # Create an output dict for final clusters
    clusters = {}

    # First we get the overlapping nodes and clusters
    overlapping_nodes = get_overlapping_clusters(cliques)

    # Check if there are overlapping nodes
    if overlapping_nodes:
        # Check use the previous_clusters info
        if previous_clusters:
            # Get final clusters from historical information, nodes checked, new cliques (overwrite previous)
            # and if the historical information has been used
            clusters_list, nodes_checked, cliques, historical_info_used = \
                get_final_clusters_from_historical(overlapping_nodes=overlapping_nodes,
                                                   previous_clusters=previous_clusters,
                                                   instant=instant, cliques=cliques,
                                                   historical_info_used=historical_info_used,
                                                   temporal_offset=temporal_offset)

        # Get final clusters from overlapping nodes. If historical information, use previous values also
        clusters = get_final_clusters_from_overlapping_nodes(all_nodes=all_nodes, graph_nodes=graph_nodes,
                                                             cliques=cliques, nodes_checked=nodes_checked,
                                                             clusters_list=clusters_list,
                                                             instant_consistencies=instant_consistencies)

    else:
        # Get final clusters from non overlapping nodes
        clusters = get_final_clusters_from_non_overlapping_nodes(all_nodes=all_nodes, graph_nodes=graph_nodes,
                                                                 cliques=cliques)

    return clusters, overlapping_nodes, historical_info_used


def get_final_clusters_from_non_overlapping_nodes(all_nodes: list, graph_nodes: list, cliques: list[tuple]):
    """
    Get final clusters when there are not overlapping nodes

    :param all_nodes: all nodes existent
    :type all_nodes: list
    :param graph_nodes: filtered graph nodes
    :type graph_nodes: list
    :param cliques: cliques from filtered graph as tuple (mean, cliques_nodes)
    :type cliques: list[tuple]
    :return:
    """
    # Important, in these cliques there can not be repeated nodes as there are no overlapping

    # First initialize the clusters that are alone, not in the graph (which is filtered)
    clusters = {item: [item] for item in set(all_nodes) - set(graph_nodes)}

    # Iterate over the cliques, which are ordered by mean consistency
    for clique_mean, clique_nodes in cliques:
        # Check if the current clique is superset of any previous clusters except those individual
        superset_key = [key for key, cluster_nodes in clusters.items() if
                        set(clique_nodes).issuperset(set(cluster_nodes)) and len(cluster_nodes) > 1]

        if superset_key:
            # If it is a superset of any cluster, replace the previous value with the new cluster
            # Access to item [0] as it is a list
            clusters[superset_key[0]] = clique_nodes
        else:
            # Otherwise create a new cluster
            clusters[clique_nodes[0]] = clique_nodes

    return clusters


def get_final_clusters_from_historical(overlapping_nodes: dict, previous_clusters: dict, instant: int,
                                       cliques: list[tuple], historical_info_used: list, temporal_offset: int = 0):
    """
    Get final clusters when historical information is used, based on temporal binding factor

    :param overlapping_nodes: overlapping nodes and its related overlapping clusters
    :type overlapping_nodes: dict
    :param previous_clusters: previous clusters information only enabled if historical info used. Default to None
    :type previous_clusters: dict
    :param instant: current instant
    :type instant: int
    :param cliques: maximum cliques from filtered graph as tuple (mean, cliques_nodes)
    :type cliques: list[tuple]
    :param historical_info_used: list for storing instant where historical info have been used
    :type historical_info_used: list
    :param temporal_offset: offset to consider current instant on previous full pattern. If 0, do not consider it,
        just use previous instants
    :type temporal_offset: int
    :return: clusters lists, nodes checked, updated cliques and historical information
    """
    # Create a copy of the cliques
    updated_cliques = copy.deepcopy(cliques)

    # Define clusters and nodes checked lists
    clusters_list, nodes_checked = [], []

    # Calculate the temporal binding factor for each overlapping node
    temporal_binding_factor = calculate_temporal_binding_factor(overlapping_nodes=overlapping_nodes,
                                                                previous_clusters=previous_clusters)

    # Iterate over the factor to select the best value for each node
    for node, probabilities in temporal_binding_factor.items():
        selected_cluster = []
        # Check if node have been checked previously
        if node not in nodes_checked:
            # All equal (equiprobable)
            if all_equal(probabilities):
                # Check previous instant whether with temporal offset or instant value
                if temporal_offset:
                    previous_instant = instant - temporal_offset
                else:
                    previous_instant = instant - 1
                # Select previous cluster directly if all new values has the same probability
                if node in previous_clusters[previous_instant]['clusters']:
                    selected_cluster = previous_clusters[previous_instant]['clusters'][node]

            else:
                # Otherwise
                # Get the index of the highest value
                max_idx = probabilities.index(max(probabilities))
                # Get the cluster with the highest probability value
                selected_cluster = overlapping_nodes[node][max_idx]

        # Append the selected cluster to the list
        clusters_list.append(selected_cluster)

        # Append to nodes checked
        nodes_checked.extend(selected_cluster)

        # Remove from updated cliques the selected cluster as it has been added
        updated_cliques = [clique for clique in updated_cliques if clique[1] != selected_cluster]

    # Pop the checked nodes from updated cliques, creating a new list without the nodes and a mean of 0.0 as it will
    # be recalculated afterward
    updated_cliques = [(0.00, [item for item in clique[1] if item not in nodes_checked]) for clique in updated_cliques]
    # Append the instant as it has been used historical information
    historical_info_used.append(instant)

    return clusters_list, nodes_checked, updated_cliques, historical_info_used


def get_final_clusters_from_overlapping_nodes(all_nodes: list, graph_nodes: list, cliques: list[tuple],
                                              nodes_checked: list, clusters_list: list, instant_consistencies: dict):
    """
    Get final clusters from overlapping nodes

    :param all_nodes: all nodes existent
    :type all_nodes: list
    :param graph_nodes: filtered graph nodes
    :type graph_nodes: list
    :param cliques: maximum cliques from filtered graph as tuple (mean, cliques_nodes)
    :type cliques: list[tuple]
    :param nodes_checked: list of nodes that have been checked
    :type nodes_checked: list
    :param clusters_list: list of clusters already created (with value if historical information used)
    :type clusters_list: list
    :param instant_consistencies: consistencies for the given instant
    :type instant_consistencies: dict
    :return: clusters dict
    """
    # Define the clusters dict
    clusters = {}

    # Append to cliques the clusters if there are not in nodes checked. A default intra_mean of 0.00.
    if set(all_nodes) - set(graph_nodes):
        cliques.append((0.00, [item for item in set(all_nodes) - set(graph_nodes) if item not in nodes_checked]))

    # Sort and recalculate the cluster by its average intra_mean
    sorted_cliques_tuples = recalculate_sort_cliques(cliques, instant_consistencies)

    # Iterate until all nodes have been checked or there are no more cliques
    while nodes_checked != graph_nodes and sorted_cliques_tuples:
        # Get the clique that is greater than single item
        clique, clique_idx = get_min_length_cliques(sorted_cliques_tuples)

        # If there are no clique greater than a single item
        if not clique:
            # Get those nodes from clique that have not been checked previously. Get first element
            clique = [node for node in sorted_cliques_tuples[0][1] if node not in nodes_checked]
            # Remove from sorted cliques the first element
            sorted_cliques_tuples.pop(0)

        else:
            # Pop clique with more than one single item
            sorted_cliques_tuples.pop(clique_idx)

        # Check if clique is a subset of nodes_checked, meaning it has been stored previously
        if clique and not set(clique).issubset(nodes_checked):
            # Append nodes checked from the clique
            nodes_checked += clique
            # Add clique to clusters list
            clusters_list.append(clique)

            # Calculate and sort the new cliques without the checked nodes
            for j in range(len(sorted_cliques_tuples)):
                # Remove checked values
                removed_nodes_clique = [node for node in sorted_cliques_tuples[j][1] if
                                        node not in nodes_checked]

                # If there is a clique with information
                if removed_nodes_clique:
                    # Get clique consistencies
                    clique_consistencies = get_clique_consistencies(removed_nodes_clique, instant_consistencies)
                    # Calculate the mean consistency
                    mean_consistencies = sum(clique_consistencies) / len(clique_consistencies) \
                        if len(clique_consistencies) > 0 else 0.00

                    # Calculate new tuple of mean consistency and the clique without checked nodes
                    sorted_cliques_tuples[j] = (mean_consistencies if not np.isnan(mean_consistencies) else 0.0,
                                                removed_nodes_clique)

            # Sort again by intra_mean and re-do the while
            sorted_cliques_tuples = sorted(sorted_cliques_tuples, key=lambda x: x[0])

    # Store clusters as dict, instead of list
    for cluster in clusters_list:
        if cluster:
            clusters[cluster[0]] = cluster

    return clusters
