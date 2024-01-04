from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score

from temporal_series_clustering.storage.clusters_history import ClustersHistory


class KMeansClustering:
    """
    Class representing K-means clustering algorithm
    """

    def __init__(self, clusters_history: ClustersHistory, input_time_series: list, base_vertices: list, k: int):
        self._clusters_history = clusters_history
        self._input_time_series = input_time_series
        self._base_vertices = base_vertices
        self._k = k
        # Create a KMeans instance with the desired number of clusters
        self._kmeans = KMeans(n_clusters=k, n_init=10)

    def perform_all_instants_clustering(self):
        # Iterate over all the instants, indicated by the first time series, there are all the same
        for instant in range(len(self._input_time_series[0])):
            self._perform_instant_clustering(instant=instant)

    def _perform_instant_clustering(self, instant):
        # STEP 1: Get the values and process it
        # Get the values
        instant_values = [time_serie[instant] for time_serie in self._input_time_series]

        # Convert to a 2D array
        instant_values = np.array(instant_values).reshape(-1, 1)

        # STEP 2: Fit the algorithm with the data
        # Fit with the data
        self._kmeans.fit(instant_values)

        # Get the cluster assignments for each data point
        assignments = self._kmeans.labels_

        # STEP 3: Create clusters
        # Create a dictionary to store the clusters
        clusters = {}
        # Assign each base vertex to its corresponding cluster
        for base_vertex, value, assignment in zip(self._base_vertices, instant_values, assignments):
            if assignment not in clusters:
                clusters[assignment] = []
            clusters[assignment].append(base_vertex)

        # Join all individual clusters to outliers
        final_clusters = {'outliers': {'nodes': [], 'intra_mean': 0.0}}
        labels = {'outliers': []}
        cluster_idx = 1
        for assignment, nodes in clusters.items():
            if len(nodes) == 1:
                final_clusters['outliers']['nodes'].extend(nodes)
                labels['outliers'].append(assignment)
            else:
                cluster_name = f'cluster_{cluster_idx}'
                if cluster_name not in final_clusters:
                    final_clusters[cluster_name] = {'nodes': []}
                final_clusters[cluster_name]['nodes'].extend(nodes)
                labels[cluster_name] = [assignment]
                cluster_idx += 1

        # STEP 4: CLUSTERS ADDITIONAL INFO
        # Define clusters info
        instant_clusters_info = {"value": {}, "intra_mean": 0.0, "inter_mean": 0.0, "silhouette_score": 0.0}

        # Insert individual cluster intra-mean
        for cluster_name, cluster_nodes in final_clusters.items():
            # Calculate intra-cluster mean (with those nodes belonging to same cluster)
            intra_cluster_mean = np.mean([np.mean(instant_values[self._kmeans.labels_ == i])
                                          for i in labels[cluster_name]])

            instant_clusters_info["value"][cluster_name] = {'nodes': cluster_nodes['nodes'],
                                                            'intra_mean': intra_cluster_mean}

        # Insert instant intra-mean
        instant_clusters_info['intra_mean'] = np.mean([v['intra_mean']
                                                       for k, v in instant_clusters_info['value'].items()])

        # Insert instant inter-mean (with those nodes belonging to different clusters that are not themselves)
        instant_clusters_info['inter_mean'] = np.mean([np.mean(instant_values[self._kmeans.labels_ != i])
                                                       for i in range(self._kmeans.n_clusters)])

        # Only calculate silhouette if there is more than one single cluster, not counting outliers
        if len(list(instant_clusters_info["value"].keys())) > 2:
            # Insert instant silhouette score
            instant_clusters_info['silhouette'] = silhouette_score(instant_values, self._kmeans.labels_)
        else:
            instant_clusters_info['silhouette'] = 0.0
        # Insert cluster intra means
        self._clusters_history.insert_all_info_on_instant(instant, instant_clusters_info)

    @property
    def clusters_history(self):
        """Getter for '_clusters_history'."""
        return self._clusters_history

    @clusters_history.setter
    def clusters_history(self, value):
        """Setter for '_clusters_history'."""
        self._clusters_history = value
