import json
import os

from temporal_series_clustering.static.plots import *
from temporal_series_clustering.storage.epsilons import EpsilonValues

if __name__ == "__main__":
    show_plots = False

    path = '../results/plots/'

    exists = os.path.exists(path)
    if not exists:
        # Create a new directory because it does not exist
        os.makedirs(path)

    with open('../results/experiment.json') as f:
        d = json.load(f)

    # Create a Epsilon Values storage
    epsilon_values = EpsilonValues()

    epsilon_values.info = d

    possible_epsilons = list(epsilon_values.info.keys())

    show_avg_all_instants_metric_against_epsilon(metric="intra_mean", epsilon_values=epsilon_values,
                                                 show_plots=show_plots)

    show_avg_all_instants_metric_against_epsilon(metric="min_inter_mean", epsilon_values=epsilon_values,
                                                 show_plots=show_plots)

    show_metric_against_time(metric="intra_mean", epsilon_values=epsilon_values, show_plots=show_plots)

    show_metric_against_time(metric="min_inter_mean", epsilon_values=epsilon_values, show_plots=show_plots)

    for epsilon in possible_epsilons:
        show_metric_against_time_by_epsilon(metric="intra_mean", epsilon=epsilon, epsilon_values=epsilon_values,
                                            show_plots=show_plots)

        show_metric_against_time_by_epsilon(metric="min_inter_mean", epsilon=epsilon, epsilon_values=epsilon_values,
                                            show_plots=show_plots)

    show_avg_num_clusters_against_epsilon(epsilon_values=epsilon_values, show_plots=show_plots)

    show_num_clusters_against_time(epsilon_values=epsilon_values, show_plots=show_plots)

    for epsilon in possible_epsilons:
        show_global_metric_against_time_by_epsilon(metric="intra_mean", epsilon=epsilon, epsilon_values=epsilon_values,
                                                   show_plots=show_plots)

        show_global_metric_against_time_by_epsilon(metric="min_inter_mean", epsilon=epsilon,
                                                   epsilon_values=epsilon_values,
                                                   show_plots=show_plots)