from ordered_set import OrderedSet

from temporal_series_clustering.cluster.graph_utils import get_cycles_info_from_graph, get_clusters_from_cycles, \
    filter_edges_by_epsilon
from temporal_series_clustering.cluster.cluster_utils import rename_clusters
from temporal_series_clustering.storage.clusters_history import ClustersHistory
from temporal_series_clustering.storage.consistencies import ConsistenciesHistory
from temporal_series_clustering.storage.simplified_graphs import SimplifiedGraphsHistory


class TCBC:
    """
    Class representing the Temporal Consistency-based Clustering algorithm

    """

    def __init__(self, clusters_history: ClustersHistory, consistencies_history: ConsistenciesHistory,
                 simplified_graphs_history: SimplifiedGraphsHistory, base_vertices: list, epsilon: float,
                 use_historical: list, temporal_window: int = 5):
        self._clusters_history = clusters_history
        self._consistencies_history = consistencies_history
        self._simplified_graphs_history = simplified_graphs_history
        self._base_vertices = base_vertices
        self._epsilon = epsilon
        self._use_historical = use_historical
        self._temporal_window = temporal_window

    def perform_all_instants_clustering(self):
        historical_info_used = []
        # Iterate over all the instants
        for instant, instant_consistencies in self._consistencies_history.info.items():
            self._perform_instant_clustering(instant=instant, instant_consistencies=instant_consistencies,
                                             historical_info_used=historical_info_used)

        return OrderedSet(historical_info_used)

    def _perform_instant_clustering(self, instant: int, instant_consistencies: dict, historical_info_used: list):
        """

        :param instant:
        :param instant_consistencies:
        :param historical_info_used:
        :return:
        """

        # Retrieve the simplified graph for the instant
        simplified_graph = self._simplified_graphs_history.get_simplified_graph_on_instant(instant)

        # Retrieve the previous instant cluster
        previous_clusters = self.get_previous_clusters_info(instant)

        # STEP 1: CREATE SIMPLIFIED GRAPH
        # From the graph, create a simplified version where there are only store those nodes lower than epsilon
        filtered_G = filter_edges_by_epsilon(simplified_graph, self._epsilon)

        # STEP 2: CYCLES
        # Get the cycles of the graph with its mean consistency value
        sorted_cycles = get_cycles_info_from_graph(filtered_G, instant_consistencies)

        # STEP 3: CLUSTERS
        instant_clusters, conflicting_clusters, historical_info_instant_used = \
            get_clusters_from_cycles(base_vertices=self._base_vertices, cliques=sorted_cycles,
                                     graph_nodes=filtered_G.nodes, instant=instant,
                                     instant_consistencies=instant_consistencies, previous_clusters=previous_clusters)

        # Add the instants on which the historical information has been used
        historical_info_used.extend(historical_info_instant_used)

        # Rename the clusters
        instant_clusters = rename_clusters(instant_clusters)

        # STEP 4: CLUSTERS ADDITIONAL INFO
        # Firs it is required to create the clusters
        self._clusters_history.insert_cluster_metrics(instant=instant, clusters=instant_clusters,
                                                      instant_consistencies=instant_consistencies)

        # Insert conflicting clusters
        self._clusters_history.insert_conflicting_clusters(instant=instant, conflicting_clusters=conflicting_clusters)

        # Calculate the intra cluster mean
        self._clusters_history.calculate_instant_intra_mean(instant)

        # Calculate the inter cluster mean (average and min) with the simplified (not filtered) graph
        self._clusters_history.calculate_instant_inter_mean(instant, all_nodes=simplified_graph.nodes,
                                                            instant_consistencies=instant_consistencies)

        # Calculate the silhouette score
        self._clusters_history.calculate_instant_silhouette(instant, instant_consistencies=instant_consistencies)

    def get_previous_clusters_info(self, instant):
        previous_clusters = {}
        # For the specified temporal window
        for i in range(1, self._temporal_window + 1):
            # Only store valid values
            if instant - i >= 0:
                clusters = {}
                for cluster, nodes_in_cluster in self._clusters_history.get_cluster_nodes_on_instant(
                        instant - i).items():
                    # Iterate over the nodes in the cluster
                    for node in nodes_in_cluster:
                        # Add the node to the result dictionary with the list of nodes in its cluster
                        clusters[node] = [n for n in nodes_in_cluster]

                # store the clusters
                previous_clusters[instant - i] = {'clusters': clusters,
                                                  'conflicting_clusters':
                                                      self._clusters_history.get_conflicting_clusters_on_instant(
                                                          instant - i)}

        # Sort the history
        previous_clusters = dict(sorted(previous_clusters.items()))

        if instant not in self._use_historical:
            previous_clusters = None

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
    def base_vertices(self):
        """Getter for '_base_vertices'."""
        return self._base_vertices

    @base_vertices.setter
    def base_vertices(self, value):
        """Setter for '_base_vertices'."""
        self._base_vertices = value

    @property
    def epsilon(self):
        """Getter for '_epsilon'."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        """Setter for '_epsilon'."""
        self._epsilon = value
