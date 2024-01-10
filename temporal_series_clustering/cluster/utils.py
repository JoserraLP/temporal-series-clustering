def find_key_of_item(item, dictionary):
    for key, value in dictionary.items():
        if any(item in sublist for sublist in value):
            return key
    return None
