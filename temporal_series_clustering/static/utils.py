from pyvis.network import Network


def show_graph(graph):
    nt = Network()
    # populates the nodes and edges data structures
    nt.from_nx(graph)
    nt.show('nx.html', notebook=False)


def process_clusters(clusters: dict, consistencies: dict):
    clusters_data = {'instant': {}}

    # Also append to clusters the predictions and the consistencies
    for k, v in clusters.items():
        clusters_data['instant'][k] = {'clusters': v,
                                       'consistencies': dict(sorted(consistencies[k].items()))}

    # Retrieve the intra mean values for each clusters
    intra_mean_clusters_values = [v['clusters']['intra_mean'] for k, v in clusters_data['instant'].items()
                                  if v['clusters']['intra_mean']]

    # Calculate the overall mean for all the time
    clusters_data['intra_mean'] = sum(intra_mean_clusters_values) / len(intra_mean_clusters_values) \
        if len(intra_mean_clusters_values) > 0 else 0.0

    # Retrieve the inter mean values for each clusters
    inter_mean_clusters_values = [v['clusters']['inter_mean'] for k, v in clusters_data['instant'].items()
                                  if v['clusters']['inter_mean']]

    # Calculate the overall mean for all the time
    clusters_data['inter_mean'] = sum(inter_mean_clusters_values) / len(inter_mean_clusters_values) \
        if len(inter_mean_clusters_values) > 0 else 0.0

    """
    # Retrieve the silhouette values for each clusters
    silhouette_clusters_values = [v['clusters']['silhouette_score'] for k, v in clusters_data['instant'].items()
                                  if v['clusters']['silhouette_score']]

    # Calculate the overall mean for all the time
    clusters_data['silhouette_score'] = sum(silhouette_clusters_values) / len(silhouette_clusters_values) \
        if len(silhouette_clusters_values) > 0 else 0.0
    """
    return clusters_data
