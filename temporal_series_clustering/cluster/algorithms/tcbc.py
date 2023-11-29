import time

from ordered_set import OrderedSet

from temporal_series_clustering.cluster.graph_utils import get_cycles_info_from_graph, get_clusters_from_cycles, \
    filter_edges_by_epsilon
from temporal_series_clustering.cluster.utils import create_outliers_cluster, rename_clusters
from temporal_series_clustering.storage.clusters_history import ClustersHistory
from temporal_series_clustering.storage.consistencies import ConsistenciesHistory
from temporal_series_clustering.storage.simplified_graphs import SimplifiedGraphsHistory


class TCBC:
    """
    Class representing the Temporal Consistency-based Clustering algorithm

    """

    def __init__(self, clusters_history: ClustersHistory, consistencies_history: ConsistenciesHistory,
                 simplified_graphs_history: SimplifiedGraphsHistory, base_vertices: list, epsilon: float):
        self._clusters_history = clusters_history
        self._consistencies_history = consistencies_history
        self._simplified_graphs_history = simplified_graphs_history
        self._base_vertices = base_vertices
        self._epsilon = epsilon

    def perform_all_instants_clustering(self):
        historical_info_used = []
        start_time = time.time()
        # Iterate over all the instants
        for instant, instant_consistencies in self._consistencies_history.info.items():
            self._perform_instant_clustering(instant=instant, instant_consistencies=instant_consistencies,
                                             historical_info_used=historical_info_used)

        print(f"DFS lasted time {time.time() - start_time}")

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
        previous_clusters_nodes = self._clusters_history.get_cluster_nodes_on_instant(instant - 1)

        # STEP 1: CREATE SIMPLIFIED GRAPH
        # From the graph, create a simplified version where there are only store those nodes lower than epsilon
        filtered_G = filter_edges_by_epsilon(simplified_graph, self._epsilon)

        # STEP 2: CYCLES
        # Get the cycles of the graph with its mean consistency value
        sorted_cycles = get_cycles_info_from_graph(filtered_G, instant_consistencies)

        # STEP 3: CLUSTERS
        # Get all the clusters from the cycles and if the historical information has been used
        instant_clusters, historical_info_instant_used = get_clusters_from_cycles(base_vertices=self._base_vertices,
                                                                                  cycles=sorted_cycles,
                                                                                  graph_nodes=filtered_G.nodes,
                                                                                  instant=instant,
                                                                                  instant_consistencies=
                                                                                  instant_consistencies,
                                                                                  previous_clusters_nodes=
                                                                                  previous_clusters_nodes)

        # Add the instants on which the historical information has been used
        historical_info_used.extend(historical_info_instant_used)

        # Create outliers cluster from individual nodes
        instant_clusters = create_outliers_cluster(instant_clusters)

        # Rename the clusters
        instant_clusters = rename_clusters(instant_clusters)


        # STEP 4: CLUSTERS ADDITIONAL INFO
        # Firs it is required to create the clusters
        self._clusters_history.insert_cluster_metrics(instant=instant, clusters=instant_clusters,
                                                      instant_consistencies=instant_consistencies)

        # Calculate the intra cluster mean
        self._clusters_history.calculate_instant_intra_mean(instant)

        # Calculate the inter cluster mean (average and min) with the simplified (not filtered) graph
        self._clusters_history.calculate_instant_inter_mean(instant, all_nodes=simplified_graph.nodes,
                                                            instant_consistencies=instant_consistencies)

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
