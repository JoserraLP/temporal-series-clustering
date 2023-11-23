import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN

from temporal_series_clustering.cluster.algorithms.mean_cycle import get_cycle_consistencies


def dbscan_cluster_graph(G, **kwargs):
    # Get the adjacency matrix (A) from the graph
    A = nx.adjacency_matrix(G, weight='consistency')

    epsilon = kwargs['epsilon']
    consistency_dict = kwargs['consistency']

    # Apply DBSCAN on the adjacency matrix.
    # Note: In this example, we're using the weights from the graph as our data points for DBSCAN.
    # You might want to adjust the parameters (eps and min_samples) based on your specific use case.
    db = DBSCAN(eps=epsilon, min_samples=1).fit(A)

    # Get the labels (clusters) from the DBSCAN output
    labels = db.labels_

    # Create a dictionary that maps each node to a cluster
    node_to_cluster = {node: cluster for node, cluster in zip(G.nodes, labels)}

    # Create an empty dictionary for the clusters
    clusters = {}

    # Iterate over the items in the original dictionary
    for key, value in node_to_cluster.items():
        # If the value is not already a key in the clusters, add it with an empty list as its value
        if value not in clusters:
            clusters[value] = []
        # Append the original key to the list of values for this key in the clusters
        clusters[value].append(key)

    # Now, if you want the keys to be the first element of each list, you can do this:
    clusters = {v[0]: v for k, v in clusters.items()}

    # Define the clusters info dict
    clusters_info = {'value': {}, 'mean': None}

    # Iterate over the clusters to get the mean consistency for a given instant
    for k, cluster in clusters.items():
        # Get consistencies of the cluster
        cluster_consistencies = get_cycle_consistencies(cluster, consistency_dict)
        # Calculate the mean value of the consistency. If not valid set infinite as we search the minimum
        mean_consistency_cluster = sum(cluster_consistencies) / len(cluster_consistencies) if cluster_consistencies \
            else None
        # Append data
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


def dbscan_cluster(epsilon, input_time_series, base_vertices):
    # input_time_series is a list of lists
    # Iterate over the series items (instants)

    dbscan_clusters = {"instant": {}, "mean": None}

    # Initialize dbscan algorithm
    db = DBSCAN(eps=epsilon, min_samples=1)

    for instant in range(len(input_time_series[0])):
        values = [time_serie[instant] for time_serie in input_time_series]

        # Convert to a 2D array
        values = np.array(values).reshape(-1, 1)

        # Apply DBSCAN on the input value
        db.fit(values)

        # Get the labels (clusters) from the DBSCAN output
        labels = db.labels_

        # Create an empty dictionary for the clusters
        clusters = {}

        # Assign each base vertex to its corresponding cluster
        for base_vertex, value, assignment in zip(base_vertices, values, labels):
            if assignment not in clusters:
                clusters[assignment] = {'vertices': [], 'values': []}
            clusters[assignment]['vertices'].append(base_vertex)
            clusters[assignment]['values'].append(value[0])

        for cluster, v in clusters.items():
            if instant not in dbscan_clusters["instant"]:
                dbscan_clusters["instant"][instant] = {"clusters": {"value": {}, "mean": None}}

            # Mean is the same as the sheaf model (resumed as std)
            dbscan_clusters["instant"][instant]["clusters"]["value"][cluster] = {"nodes": v['vertices'],
                                                                                 "mean":
                                                                                     np.std(v['values'])}

        # Retrieve the mean values for each clusters
        mean_clusters_values = [v['mean'] for k, v in
                                dbscan_clusters["instant"][instant]["clusters"]["value"].items()
                                if v['mean']]

        dbscan_clusters["instant"][instant]["clusters"]["mean"] = sum(mean_clusters_values) / len(
            mean_clusters_values) if len(mean_clusters_values) > 0 else 0.0

    mean_clusters_all_instants = [v for instant in range(len(input_time_series[0]))
                                  for k, v in dbscan_clusters["instant"][instant]["clusters"].items() if
                                  k == 'mean']

    # Remove None values
    mean_clusters_all_instants = [x for x in mean_clusters_all_instants if x is not None]

    dbscan_clusters["mean"] = sum(mean_clusters_all_instants) / len(mean_clusters_all_instants) if len(
        mean_clusters_all_instants) > 0 \
        else 0.0

    return dbscan_clusters
