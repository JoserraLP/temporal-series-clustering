import copy

import numpy as np
from sklearn.metrics import silhouette_score

from temporal_series_clustering.cluster.graph_utils import get_clique_consistencies


class ClustersHistory:
    """
    Class storing the historical clusters of the algorithms.

    Structure of each instant is:
     - "value": storing the representative node and both the nodes belong to each cluster and its intra_mean e.g.

     "value": {"B": {"nodes": ["B"], "intra_mean": 0.0}, "C": {"nodes": ["C","A"], "intra_mean": 0.007071067811865474}}}

     - "intra_mean": storing the mean of the intra_mean of all the clusters for the given instant

     - "inter_mean": storing the inter mean of the combination of all those nodes not belonging to the same cluster for
     the given instant
    """

    def __init__(self):
        # In this dictionary all the information per each instant will be stored
        self._info = {}

    def insert_all_info_on_instant(self, instant: int, instant_info: dict):
        """
        Insert/update the information on a given instant

        :param instant: instant to store the information
        :type instant: int
        :param instant_info: instant information to store
        :type instant_info: dict
        :return: None
        """
        self._info[instant] = instant_info

    def insert_cluster_metrics(self, clusters: dict, instant_consistencies: dict, instant: int):
        """
        Store the cluster metrics as the intra_mean

        :param clusters: dict with clusters information
        :type clusters: dict
        :param instant: instant to insert the information
        :type instant: int
        :param instant_consistencies: consistencies for the given instant
        :type instant_consistencies: dict
        :return: None
        """
        # If instant has not been created, initialize it
        if instant not in self._info:
            self._info[instant] = {'value': {}}

        # Iterate over the clusters nodes to get the mean consistency for a given instant
        for cluster_id, cluster in clusters.items():
            # Get consistencies of the cluster
            cluster_consistencies = get_clique_consistencies(cluster, instant_consistencies)

            # Calculate the intra_mean value of the consistency. If not valid set 0.0
            intra_mean_consistency_cluster = sum(cluster_consistencies) / len(cluster_consistencies) \
                if cluster_consistencies else 0.0

            # Insert into the historical info (nodes and intra_mean)
            self._info[instant]['value'][cluster_id] = {
                'nodes': cluster,
                'intra_mean': intra_mean_consistency_cluster
            }

    def calculate_instant_intra_mean(self, instant: int):
        """
        Calculate the intra_mean of the clusters for a given instant. Default to 0.0

        :param instant: instant to calculate the intra_mean
        :type instant: int
        :return: None
        """
        if instant in self._info:
            # Retrieve the mean values for each clusters
            intra_mean_clusters_values = [v['intra_mean'] for k, v in self._info[instant]['value'].items()
                                          if v['intra_mean']]

            # Store the overall intra mean of a given instant
            self._info[instant]['intra_mean'] = sum(intra_mean_clusters_values) / len(intra_mean_clusters_values) \
                if len(intra_mean_clusters_values) > 0 else 0.0

    def calculate_instant_inter_mean(self, instant: int, all_nodes: list, instant_consistencies: dict) -> None:
        """
        Calculate the inter_mean of the clusters for a given instant. Default 0.0

        :param instant: instant to calculate the inter_mean
        :type instant: int
        :param all_nodes: all the nodes of the full graph
        :type all_nodes: list
        :param instant_consistencies: consistencies for the given instant
        :type instant_consistencies: dict
        :return: None
        """

        if instant in self._info:
            # Initialize a variable to sum up all the consistencies
            inter_consistencies_values = []

            # Initialize a list for those connections checked
            connections_checked = []
            # Iterate over all the nodes with the full graph, not the filtered
            for node in all_nodes:
                # Create a list for those nodes belonging to the same cluster
                nodes_same_cluster = []

                # Retrieve the cluster it belongs to
                for k, cluster in self._info[instant]['value'].items():
                    # Cluster items are the 'nodes' key
                    cluster = cluster['nodes']
                    if node in cluster:
                        nodes_same_cluster = copy.deepcopy(cluster)
                        if node in nodes_same_cluster:
                            nodes_same_cluster.remove(node)

                # Retrieve all those nodes that are not in the same cluster
                inter_nodes = [item for item in all_nodes if item != node and item not in nodes_same_cluster]
                # Iterate over the nodes
                for inter_node in inter_nodes:
                    # Get keys for both possible directions
                    key1 = node + '_' + inter_node
                    key2 = inter_node + '_' + node
                    # Store the value that exists and if it's not stored previously
                    if key1 in instant_consistencies and key1 not in connections_checked:
                        # Append consistencies value
                        inter_consistencies_values.append(instant_consistencies[key1])
                        # Append the connection as it has been checked
                        connections_checked.append(key1)
                    elif key2 in instant_consistencies and key2 not in connections_checked:
                        # Append consistencies value
                        inter_consistencies_values.append(instant_consistencies[key2])
                        # Append the connection as it has been checked
                        connections_checked.append(key2)

            # Calculate the overall inter mean of a given instant
            self._info[instant]['inter_mean'] = sum(inter_consistencies_values) / len(inter_consistencies_values) \
                if len(inter_consistencies_values) > 0 else 0.0

            # Instead of mean, get the minimum
            self._info[instant]['min_inter_mean'] = min(inter_consistencies_values) \
                if len(inter_consistencies_values) > 0 else 0.0

    def calculate_instant_silhouette(self, instant: int, instant_consistencies: dict):
        """
        Calculate silhouette score based on consistencies for a given instant. Default to 0.0

        :param instant: instant to calculate the silhouette score
        :type instant: int
        :param instant_consistencies: consistencies for the given instant
        :type instant_consistencies: dict
        :return: None
        """
        # Calculate the silhouette score
        nodes = sorted(set([node for k, nodes_cluster in self._info[instant]['value'].items()
                            for node in nodes_cluster['nodes']]))

        # Create a distance matrix
        distance_matrix = np.zeros((len(nodes), len(nodes)))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j:
                    key = '_'.join([node_i, node_j])
                    # Alternative if it does not exist the key
                    if key in instant_consistencies:
                        distance_matrix[i, j] = instant_consistencies[key]
                    else:
                        distance_matrix[i, j] = instant_consistencies['_'.join([node_j, node_i])]

        # Retrieve the clusters
        clusters = [nodes_cluster['nodes'] for k, nodes_cluster in self._info[instant]['value'].items()
                    if nodes_cluster['nodes']]

        # Create labels for each node based on clusters
        labels = np.zeros(len(nodes), dtype=int)

        for i, cluster in enumerate(clusters):
            for node in cluster:
                labels[nodes.index(node)] = i

        # Only calculate the silhouette when there is more than one cluster or is less than labels
        silhouette_score_val = silhouette_score(distance_matrix, labels, metric='precomputed') if len(
            clusters) > 1 and len(clusters) < len(labels) else 0.0

        # Calculate silhouette score
        self._info[instant]['silhouette_score'] = silhouette_score_val

    def get_all_info_on_instant(self, instant: int):
        """
        Get all info for a given instant

        :param instant: instant to retrieve all info
        :type instant: int
        :return: dict with the information if exists, None otherwise
        """
        instant_info = None
        if instant in self._info:
            instant_info = self._info[instant]

        return instant_info

    def get_cluster_nodes_on_instant(self, instant: int) -> dict:
        """
        Get cluster nodes on a given instant

        :param instant: instant to retrieve nodes
        :type instant: int
        :return: dict with cluster name and cluster nodes
        """
        nodes_info = None
        if instant in self._info:
            nodes_info = {k: v['nodes'] for k, v in self._info[instant]['value'].items()}

        return nodes_info

    def insert_overlapping_clusters(self, overlapping_clusters: list, instant: int):
        """
        Insert overlapping clusters into the storage at a given instant

        :param overlapping_clusters: overlapping clusters
        :type overlapping_clusters: list
        :param instant: instant to store the information
        :type instant: int
        :return: None
        """
        if instant in self._info:
            self._info[instant]['overlapping_clusters'] = overlapping_clusters

    def get_overlapping_clusters_on_instant(self, instant: int):
        """
        Get the overlapping nodes and clusters for a given instant

        :param instant: instant to retrieve overlapping clusters
        :type instant: int
        :return: dict with the overlapping nodes and related overlapping clusters
        """
        overlapping_clusters = None
        if instant in self._info:
            overlapping_clusters = self._info[instant]['overlapping_clusters']
        return overlapping_clusters

    def get_all_instants_mean(self, measure: str):
        """
        Get the mean of all instants defined by measure

        :param measure: type of mean. It can be 'intra_mean' or 'inter_mean'
        :type measure: str
        :return: mean of all instants for the given measure
        """
        # Retrieve the mean values for each instant on a list
        values = [v[measure] for instant, v in self._info.items() if measure in v]
        # Calculate the mean with sum and number of elements, or 0.0 if there is no information
        return sum(values) / len(values) if len(values) > 0 else 0.0

    @property
    def info(self):
        """Getter of info dict"""
        return self._info

    @info.setter
    def info(self, value):
        """Setter of info dict"""
        self._info = value
