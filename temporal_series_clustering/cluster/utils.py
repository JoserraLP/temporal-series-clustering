from itertools import groupby


def find_key_of_item(item, dictionary):
    """
    Return the key where a specific item appears on the sublist of value

    :param item: item to find
    :param dictionary: dictionary where to find the item
    :return: key or None if not found
    """
    # Iterate over the dict
    for key, value in dictionary.items():
        # If the item is in a sublist of the value
        if any(item in sublist for sublist in value):
            return key
    return None


def all_equal(iterable):
    """
    Check if all the elements of the iterable are equal

    :param iterable: iterable where to check the elements
    :return: True if all elements are equal, False otherwise
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)
