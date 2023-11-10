
import numpy as np


def propagate_sheaf_values(sheaf, predictors_output: list, reading_freq: int, num_items, base_vertices: list):
    summary = []
    filtration = {}
    mean = []
    values = {}
    # Iterate over the number of items
    for i in range(num_items):
        # If it is a reading frequency
        if i % reading_freq == 0:
            # Propagate the value on the sheaf
            sheaf.propagate({vertex: [i, predictors_output[j][i]] for j, vertex in enumerate(base_vertices)})

            # Calculate summary
            summary.append(sheaf.value)
            # Calculate the mean of the values
            # Identity methods (directly the value)
            mean.append(np.mean([predictors_output[j][i] for j in range(len(base_vertices))]))
            # Get consistency filtration
            filtration[i] = sheaf.get_consistency_filtration()
            # Initialize values
            values[i] = {}
            # Iterate over the faces to retrieve its filtration value
            for j in filtration[i]:
                values[i][j] = sheaf.get_face(j).value

        else:
            pass

    return summary, filtration, mean, values







