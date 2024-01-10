import json

from temporal_series_clustering.storage.epsilons import EpsilonValues


def count_jumps(epsilon: str, epsilon_values: EpsilonValues) -> list:
    """
    Count the number of jumps between clusters per instant for a given epsilon

    :param epsilon: epsilon value
    :type epsilon: str
    :param epsilon_values: epsilon stored clusters information
    :type epsilon_values: EpsilonValues
    :return: list with the number of jumps on each instant
    :rtype: list
    """
    # Create a list with the number of jumps per instant, as list of 0 with length = number of instants
    jump_per_instant = [0] * len(list(epsilon_values.info[epsilon]['instant'].keys()))

    # Iterate over each instant
    for instant, value in epsilon_values.info[epsilon]['instant'].items():
        # Define a list for those nodes that have been checked
        checked_nodes = []
        # Parse to int as it is used afterwards
        instant = int(instant)
        # When the instant is 0 means that there is not jumps, the baseline
        if instant != 0:
            # Get related clusters per node from previous instant information
            prev_clusters = {}
            for cluster, cluster_value in epsilon_values.info[epsilon]['instant'][str(instant - 1)]['value'].items():
                # Iterate over the nodes and store its cluster for each one of them
                for node in cluster_value['nodes']:
                    # Do not store the nodes previously checked
                    prev_clusters[node] = {
                        'nodes': [item for item in cluster_value['nodes'] if item not in checked_nodes]}

            # Get and order the current clusters from bigger to lower, in terms of number of nodes
            cur_clusters = {k: v for k, v in sorted(value['value'].items(), key=lambda item: len(item[1]['nodes']),
                                                    reverse=True)}

            # Iterate over the current clusters
            for cluster, cluster_value in cur_clusters.items():
                # Iterate over all the current nodes
                for node in cluster_value['nodes']:
                    # If the node is in previous clusters but not yet checked
                    if node in prev_clusters and node not in checked_nodes:
                        # If the previous and the current cluster nodes are not equal
                        if set(prev_clusters[node]['nodes']) != set(cluster_value['nodes']):

                            # If current cluster is a subset of the previous cluster
                            if set(cluster_value['nodes']).issubset(set(prev_clusters[node]['nodes'])):
                                # Only consider those nodes that are not individual
                                if len(cluster_value['nodes']) != 1:
                                    # Nodes from previous clusters that are not checked and not in the actual cluster
                                    remaining_nodes = [item for item in prev_clusters[node]['nodes']
                                                       if item not in cluster_value['nodes']
                                                       and item not in checked_nodes]

                                    # Add the number of remaining nodes
                                    jump_per_instant[instant] += len(remaining_nodes)
                                    # Add the previous cluster values to nodes checked
                                    checked_nodes.extend(prev_clusters[node]['nodes'])
                            # If current cluster is a superset of the previous cluster
                            elif set(cluster_value['nodes']).issuperset(set(prev_clusters[node]['nodes'])):
                                # Nodes from current clusters that are not checked and not in the previous cluster
                                remaining_nodes = [item for item in cluster_value['nodes']
                                                   if item not in prev_clusters[node]['nodes']
                                                   and item not in checked_nodes]

                                # Add the number of remaining nodes
                                jump_per_instant[instant] += len(remaining_nodes)
                                # Add the remaining nodes
                                checked_nodes.extend(remaining_nodes)
                            # Other cases when they are not strictly related
                            else:
                                # Get symmetric difference (nodes not in intersection = outer nodes)
                                # Order is not relevant
                                symmetric_difference = \
                                    set(prev_clusters[node]['nodes']).symmetric_difference(set(cluster_value['nodes']))

                                # Nodes from symmetric difference not checked
                                remaining_nodes = [item for item in symmetric_difference if item not in checked_nodes]
                                # Add the number of remaining nodes
                                jump_per_instant[instant] += len(remaining_nodes)
                                # Add the remaining nodes
                                checked_nodes.extend(remaining_nodes)

    return jump_per_instant


def create_outliers_cluster(clusters: dict) -> dict:
    """
    Gather those individual clusters into a cluster called 'outliers'

    :param clusters: clusters with individual and grouped nodes
    :type clusters: dict
    :return: dict with clusters and outliers
    :rtype: dict
    """
    resulting_cluster = {'outliers': []}
    for key, cluster in clusters.items():
        if len(cluster) == 1:
            # Individual element, store in the outliers
            resulting_cluster['outliers'].append(cluster[0])
        else:
            # Otherwise remain the same
            resulting_cluster[key] = cluster

    return resulting_cluster


def rename_clusters(clusters: dict) -> dict:
    """
    Rename the clusters to generic names

    :param clusters: dict with clusters information
    :type clusters: dict
    :return: clusters renamed
    :rtype: dict
    """
    renamed_clusters = {}
    for i, (k, v) in enumerate(clusters.items()):
        # Do not rename the outliers cluster
        if k != 'outliers':
            renamed_clusters[f'cluster_{i}'] = v
        else:
            renamed_clusters[k] = v

    return renamed_clusters


# For sets serializing to list
class SetEncoder(json.JSONEncoder):
    """
    Class for encoding JSON sets into lists
    """

    def default(self, obj):
        return list(obj)


def store_clusters_json(clusters: dict, directory: str) -> None:
    """
    Store clusters information dict into json with the SetEncoder class

    :param clusters: clusters information
    :type clusters: dict
    :param directory: directory where the output file will be stored
    :type directory: str
    :return: None
    """
    with open(directory, 'w') as f:
        json.dump(clusters, f, cls=SetEncoder)



