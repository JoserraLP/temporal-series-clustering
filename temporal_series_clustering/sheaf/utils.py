
import numpy as np


def propagate_sheaf_values(sheaf, data_sources_output: list, reading_freq: int, pattern_length: int, 
                           base_vertices: list):
    """
    Propagate the values from each sheaf to higher levels
    
    :param sheaf: sheaf model 
    :param data_sources_output: data sources information 
    :type data_sources_output: list
    :param reading_freq: reading frequency of the data sources
    :type reading_freq: int
    :param pattern_length: number of items on patterns
    :type pattern_length: int
    :param base_vertices: name of the data sources
    :type base_vertices: list
    :return: summary, consistencies, mean and values from the sheaf propagation process
    """
    # Define variables to store information
    summary, mean = [], []
    consistency, values = {}, {}
    # Iterate over the number of items
    for i in range(pattern_length):
        # If it is a reading frequency
        if i % reading_freq == 0:
            # Propagate the value on the sheaf
            sheaf.propagate({vertex: [i, data_sources_output[j][i]] for j, vertex in enumerate(base_vertices)})

            # Calculate summary
            summary.append(sheaf.value)
            # Calculate the mean of the values
            # Identity methods (directly the value)
            mean.append(np.mean([data_sources_output[j][i] for j in range(len(base_vertices))]))
            # Get consistency
            consistency[i] = sheaf.get_consistency_filtration()
            # Initialize values
            values[i] = {}
            # Iterate over the faces to retrieve its consistency value
            for j in consistency[i]:
                values[i][j] = sheaf.get_face(j).value

        else:
            pass

    return summary, consistency, mean, values







