import networkx as nx
from sklearn.cluster import DBSCAN


def dbscan_cluster(G, **kwargs):
    # Get the adjacency matrix (A) from the graph
    A = nx.adjacency_matrix(G, weight='consistency')

    epsilon = kwargs['epsilon']

    # Apply DBSCAN on the adjacency matrix.
    # Note: In this example, we're using the weights from the graph as our data points for DBSCAN.
    # You might want to adjust the parameters (eps and min_samples) based on your specific use case.
    db = DBSCAN(eps=epsilon, min_samples=1).fit(A)

    # Get the labels (clusters) from the DBSCAN output
    labels = db.labels_

    # Create a dictionary that maps each node to a cluster
    node_to_cluster = {node: cluster for node, cluster in zip(G.nodes, labels)}

    # Create an empty dictionary for the result
    result = {}

    # Iterate over the items in the original dictionary
    for key, value in node_to_cluster.items():
        # If the value is not already a key in the result, add it with an empty list as its value
        if value not in result:
            result[value] = []
        # Append the original key to the list of values for this key in the result
        result[value].append(key)

    # Now, if you want the keys to be the first element of each list, you can do this:
    result = {v[0]: v for k, v in result.items()}

    return result
