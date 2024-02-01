from ordered_set import OrderedSet

from temporal_series_clustering.cluster.graph_utils import get_cliques_info_from_graph, get_clusters_from_cliques, \
    filter_edges_by_epsilon
from temporal_series_clustering.cluster.cluster_utils import rename_clusters
from temporal_series_clustering.storage.clusters_history import ClustersHistory
from temporal_series_clustering.storage.consistencies import ConsistenciesHistory
from temporal_series_clustering.storage.simplified_graphs import SimplifiedGraphsHistory


class TCBC:
    """
    Class representing the Temporal Consistency-based Clustering algorithm

    :param clusters_history: storage for clusters information
    :param consistencies_history: storage for consistencies information
    :param values_history: storage for input sources values information
    :param simplified_graphs_history: storage for simplified graphs information
    :param all_nodes: all nodes existent in system
    :param epsilon: value threshold for overlapping nodes
    :param use_historical: list with the intervals on which apply temporal binding algorithm
    :param temporal_window: temporal window used for temporal binding algorithm
    :param temporal_offset: offset to consider current instant on previous full pattern. If 0, do not consider it,
        just use previous instants
    """

    def __init__(self, clusters_history: ClustersHistory, consistencies_history: ConsistenciesHistory,
                 values_history: ConsistenciesHistory,
                 simplified_graphs_history: SimplifiedGraphsHistory,
                 all_nodes: list, epsilon: float,
                 use_historical: list, temporal_window: int = 5, temporal_offset: int = 0):
        self._clusters_history = clusters_history
        self._consistencies_history = consistencies_history
        self._values_history = values_history
        self._simplified_graphs_history = simplified_graphs_history
        self._base_nodes = all_nodes
        self._epsilon = epsilon
        self._use_historical = use_historical
        self._temporal_window = temporal_window
        self._temporal_offset = temporal_offset

    def perform_all_instants_clustering(self) -> OrderedSet:
        """
        Perform the clustering algorithm for all the instants

        :return: set with the instants on which historical temporal binding has been used
        """
        # Define list for historical information used on temporal binding
        historical_info_used = []
        # Iterate over all the instants
        for instant, instant_consistencies in self._consistencies_history.info.items():
            # Perform instant clustering
            self._perform_instant_clustering(instant=instant, instant_consistencies=instant_consistencies,
                                             historical_info_used=historical_info_used,
                                             instant_values=self._values_history.get_all_info_on_instant(instant))
        # Return an ordered set of the historical information used
        return OrderedSet(historical_info_used)

    def _perform_instant_clustering(self, instant: int, instant_consistencies: dict, historical_info_used: list,
                                    instant_values: dict):
        """
        Perform the clustering for a given instant

        :param instant: instant
        :type instant: int
        :param instant_consistencies: consistencies for the given instant
        :type instant_consistencies: dict
        :param instant_values: input sources values for the given instant
        :type instant_values: dict
        :param historical_info_used: list with the historical information used
        :type historical_info_used: list
        :return:
        """

        # Retrieve the simplified graph for the instant
        simplified_graph = self._simplified_graphs_history.get_simplified_graph_on_instant(instant)

        # Retrieve the previous instant cluster
        previous_clusters = self.get_temporal_window_clusters_info(instant)

        # STEP 1: CREATE SIMPLIFIED GRAPH
        # From the graph, create a simplified version where there are only store those nodes lower than epsilon
        filtered_G = filter_edges_by_epsilon(simplified_graph, self._epsilon)

        # STEP 2: CLIQUES
        # Get the cliques of the graph with its mean consistency value (mean, clique)
        sorted_cliques = get_cliques_info_from_graph(filtered_G, instant_consistencies)

        # STEP 3: CLUSTERS
        instant_clusters, overlapping_clusters, historical_info_instant_used = \
            get_clusters_from_cliques(all_nodes=self._base_nodes, cliques=sorted_cliques,
                                      graph_nodes=filtered_G.nodes, instant=instant,
                                      instant_consistencies=instant_consistencies, previous_clusters=previous_clusters,
                                      temporal_offset=self._temporal_offset)

        # Add the instants on which the historical information has been used
        historical_info_used.extend(historical_info_instant_used)

        # Rename the clusters
        instant_clusters = rename_clusters(instant_clusters)

        # STEP 4: CLUSTERS ADDITIONAL INFO
        # Firs it is required to create the clusters
        self._clusters_history.insert_cluster_metrics(instant=instant, clusters=instant_clusters,
                                                      instant_consistencies=instant_consistencies,
                                                      instant_values=instant_values)

        # Insert overlapping clusters
        self._clusters_history.insert_overlapping_clusters(instant=instant, overlapping_clusters=overlapping_clusters)

        # Calculate the intra cluster mean
        self._clusters_history.calculate_instant_intra_mean(instant)

        # Calculate the inter cluster mean (average and min) with the simplified (not filtered) graph
        self._clusters_history.calculate_instant_inter_mean(instant, all_nodes=simplified_graph.nodes,
                                                            instant_consistencies=instant_consistencies)

        # Calculate the silhouette score
        self._clusters_history.calculate_instant_silhouette(instant, instant_consistencies=instant_consistencies)

    def get_temporal_window_clusters_info(self, instant: int) -> dict:
        """
        Get the temporal window information such as final and overlapping clusters

        :param instant: instant for starting the temporal window
        :type instant: int
        :return: dict with temporal window cluster information
        :rtype: dict
        """
        # If the instant is not available to use in historical set empty
        if instant not in self._use_historical:
            previous_clusters = None
        else:
            # Initialize the dict for storing previous clusters
            previous_clusters = {}
            # For the specified temporal window
            for i in range(1, self._temporal_window + 1):
                # Only store valid values (negative instants do not exists)
                previous_instant = instant - i if self._temporal_offset == 0 else instant - self._temporal_offset*i
                if previous_instant >= 0:

                    # Define a dict for final clusters for a given instant
                    final_clusters = {}
                    # Iterate over the cluster nodes
                    for cluster_nodes in \
                            list(self._clusters_history.get_cluster_nodes_on_instant(previous_instant).values()):
                        # Iterate over the nodes in the cluster
                        for node in cluster_nodes:
                            # Add the node to the result dictionary with the list of nodes in its cluster
                            final_clusters[node] = [n for n in cluster_nodes]

                    # Store both the final and previous overlapping clusters
                    previous_clusters[previous_instant] = \
                        {'clusters': final_clusters,
                         'overlapping_clusters':
                             self._clusters_history.get_overlapping_clusters_on_instant(previous_instant)}

            # Sort the history by instant
            previous_clusters = dict(sorted(previous_clusters.items()))

        return previous_clusters

    @property
    def clusters_history(self):
        """Getter for '_clusters_history'."""
        return self._clusters_history

    @clusters_history.setter
    def clusters_history(self, value):
        """Setter for '_clusters_history'."""
        self._clusters_history = value

    @property
    def consistencies_history(self):
        """Getter for '_consistencies_history'."""
        return self._consistencies_history

    @consistencies_history.setter
    def consistencies_history(self, value):
        """Setter for '_consistencies_history'."""
        self._consistencies_history = value

    @property
    def simplified_graphs_history(self):
        """Getter for '_simplified_graphs_history'."""
        return self._simplified_graphs_history

    @simplified_graphs_history.setter
    def simplified_graphs_history(self, value):
        """Setter for '_simplified_graphs_history'."""
        self._simplified_graphs_history = value

    @property
    def base_nodes(self):
        """Getter for '_base_nodes'."""
        return self._base_nodes

    @base_nodes.setter
    def base_nodes(self, value):
        """Setter for '_base_nodes'."""
        self._base_nodes = value

    @property
    def epsilon(self):
        """Getter for '_epsilon'."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        """Setter for '_epsilon'."""
        self._epsilon = value
