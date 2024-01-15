import json
import numpy as np
import pandas as pd

from temporal_series_clustering.cluster.cluster_utils import count_jumps
from temporal_series_clustering.storage.epsilons import EpsilonValues


def load_output_clusters_file(output_clusters_dir: str) -> EpsilonValues:
    """
    Load output clusters file into a EpsilonValue object

    :param output_clusters_dir: file where the output clusters are stored
    :type output_clusters_dir: str
    :return: EpsilonValues with loaded data
    """
    with open(output_clusters_dir) as f:
        values = json.load(f)

    # Create a Epsilon Values storage
    epsilon_values = EpsilonValues()

    epsilon_values.info = values

    return epsilon_values


def get_avg_jumps(epsilon_values: EpsilonValues) -> dict:
    """
    Get the average number of jumps per epsilon

    :param epsilon_values: storage of all information
    :type epsilon_values: EpsilonValues
    :return: dictionary with the average number of jumps per epsilon
    """
    return {k: np.mean(count_jumps(k, epsilon_values)) for k, v in epsilon_values.info.items()}


def get_jumps_epsilon(epsilon_values: EpsilonValues, epsilon: str) -> dict:
    """
    Get the number of jumps per epsilon

    :param epsilon_values: storage of all information
    :type epsilon_values: EpsilonValues
    :param epsilon: epsilon value
    :type epsilon: str
    :return: dictionary with the number of jumps per epsilon
    """
    return {k: count_jumps(k, epsilon_values) for k, v in epsilon_values.info.items() if k == epsilon}


def inverse_df_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform the inverse of each item on the dataframe

    :param df: input dataframe
    :type df: pd.DataFrame
    :return: inverse items dataframe
    :rtype: pd.DataFrame
    """
    return df.map(lambda x: 1 / x if x != 0 else np.inf)


def calculate_cost_function(output_clusters_dir: str) -> pd.DataFrame:
    """
    Calculate the cost function of the output clusters specified file

    f = std(avg_jumps) - std(avg_silhouette_score)

    :param output_clusters_dir: file where it is stored the output clusters
    :type output_clusters_dir: str
    :return: DataFrame with average number of jumps, average silhouette score and the function cost
    :rtype: pd.DataFrame
    """
    epsilon_values = load_output_clusters_file(output_clusters_dir)

    # Create a dataframe
    df_info = pd.DataFrame()

    # Append average number of jumps
    df_info['avg_jumps'] = get_avg_jumps(epsilon_values)

    # Append average number of clusters
    df_info['avg_num_clusters'] = epsilon_values.get_avg_num_clusters()

    # Append silhouette score
    df_info['avg_silhouette_score'] = epsilon_values.get_avg_silhouette_score()

    # Standarize all columns
    df_info = df_info.apply(lambda x: (x - x.mean()) / x.std())

    # Define a factor alpha for weight of historical information
    alpha = 0.5

    # Calculate cost function
    # Plus as we want to minimize the average number of jumps
    # Minus as we want to maximize the average silhouette score
    df_info['f'] = alpha*df_info['avg_jumps'] - (1-alpha)*df_info['avg_silhouette_score']
    # avg_num_clusters removed -> lower epsilons will have lower cost function as they maximize num of clusters always
    # - df_info['avg_num_clusters']

    # Remove first epsilon or set to 0
    df_info['f'].iloc[0] = 0

    return df_info
