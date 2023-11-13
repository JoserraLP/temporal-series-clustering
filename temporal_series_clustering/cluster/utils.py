import json
from collections import defaultdict


# For sets serializing to list
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        return list(obj)


def store_clusters_json(clusters: dict, directory: str):
    with open(directory, 'w') as f:
        json.dump(clusters, f, cls=SetEncoder)


def gather_subsets(cluster_dict):
    # Initialize an empty dictionary to store the subsets
    subset_dict = {}

    # Iterate over each item in the cluster dictionary
    for key1, value1 in cluster_dict.items():
        # Initialize a list to store the subsets for the current key
        subsets = []
        # Iterate over each item in the cluster dictionary again
        for key2, value2 in cluster_dict.items():
            # Check if the current value is a subset of another value
            if key1 != key2 and set(value1).issubset(set(value2)):
                # If it is, append it to the subsets list
                subsets.append(key2)
        # Add the subsets list to the subset dictionary
        subset_dict[key1] = subsets

    return subset_dict


def find_common_items(cluster_dict):
    # Initialize a dictionary to store the counts
    counts = defaultdict(lambda: {'cluster': [], 'count': 0})

    # Iterate over each value in the cluster dictionary
    for cluster, value in cluster_dict.items():
        # Iterate over each item in the current value
        for item in value:
            # Increment its count
            counts[item]['count'] += 1
            counts[item]['cluster'].append(cluster)

    # Find the items that appear in at least two values
    common_items = {k: v['cluster'] for k, v in counts.items() if v['count'] >= 2}

    return common_items


def find_key_of_item(item, dictionary):
    for key, value in dictionary.items():
        if any(item in sublist for sublist in value):
            return key
    return None
