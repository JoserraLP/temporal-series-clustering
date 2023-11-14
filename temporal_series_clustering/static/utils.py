from pyvis.network import Network


def show_graph(graph):
    nt = Network()
    # populates the nodes and edges data structures
    nt.from_nx(graph)
    nt.show('nx.html', notebook=False)


def process_clusters(clusters: dict, predictions: list, consistencies: dict):
    clusters_data = {'instant': {}, 'mean': None}

    # Also append to clusters the predictions and the consistencies
    for k, v in clusters.items():
        clusters_data['instant'][k] = {'clusters': v,
                                       'predictions': [predictor[k] for predictor in predictions],
                                       'consistencies': dict(sorted(consistencies[k].items()))}

    # Retrieve the mean values for each clusters
    mean_clusters_values = [v['clusters']['mean'] for k, v in clusters_data['instant'].items() if v['clusters']['mean']]

    # Calculate the overall mean for all the time
    clusters_data['mean'] = sum(mean_clusters_values) / len(mean_clusters_values) if len(mean_clusters_values) > 0 \
        else None

    return clusters_data
