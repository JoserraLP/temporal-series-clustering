import os
import json

from temporal_series_clustering.visualization.plots import *


def show_tcbc_experiment_metrics(historical: bool, num_sources: int):
    """
    Show different metrics related to the TCBC experiments

    :param historical: use historical approach or not
    :type historical: bool
    :param num_sources: number of sources of experiment
    :type num_sources: int
    :return:
    """
    show_plots = True

    historical = 'no_' if not historical else ''

    path = f'../results/plots/{num_sources}/{historical}historical/'

    exists = os.path.exists(path)
    if not exists:
        # Create a new directory because it does not exist
        os.makedirs(path)

    with open(f'../results/experiment_{num_sources}_sources_{historical}historical.json') as f:
        d = json.load(f)

    # Create a Epsilon Values storage
    epsilon_values = EpsilonValues()

    epsilon_values.info = d

    show_avg_num_clusters(epsilon_values)

    show_avg_all_instants_metric_against_epsilon(metric="intra_mean", epsilon_values=epsilon_values,
                                                 show_plots=show_plots, results_dir=path)

    show_avg_all_instants_metric_against_epsilon(metric="min_inter_mean", epsilon_values=epsilon_values,
                                                 show_plots=show_plots, results_dir=path)

    show_metric_against_time(metric="intra_mean", epsilon_values=epsilon_values, show_plots=show_plots,
                             results_dir=path)

    show_metric_against_time(metric="min_inter_mean", epsilon_values=epsilon_values, show_plots=show_plots,
                             results_dir=path)

    for epsilon in list(epsilon_values.info.keys()):
        show_metric_against_time_by_epsilon(metric="intra_mean", epsilon=epsilon, epsilon_values=epsilon_values,
                                            show_plots=show_plots, results_dir=path)

        show_metric_against_time_by_epsilon(metric="min_inter_mean", epsilon=epsilon, epsilon_values=epsilon_values,
                                            show_plots=show_plots, results_dir=path)

    show_avg_num_clusters_against_epsilon(epsilon_values=epsilon_values, show_plots=show_plots, results_dir=path)

    show_num_clusters_against_time(epsilon_values=epsilon_values, show_plots=show_plots, results_dir=path)

    for epsilon in list(epsilon_values.info.keys()):
        show_global_metric_against_time_by_epsilon(metric="intra_mean", epsilon=epsilon, epsilon_values=epsilon_values,
                                                   show_plots=show_plots, results_dir=path)

        show_global_metric_against_time_by_epsilon(metric="min_inter_mean", epsilon=epsilon,
                                                   epsilon_values=epsilon_values,
                                                   show_plots=show_plots, results_dir=path)


if __name__ == "__main__":
    compare_tcbc_avg_num_jumps(21)

    # compare_tcbc_num_jumps_epsilon(21, "0.08")

    # compare_tcbc_num_clusters_coherence(21)

    plot_cost_function_tcbc(21)  # On historical, the best epsilon is 0.11

    # show_dbscan_avg_jumps(21)

    # show_kmeans_avg_jumps(21)

    # compare_tcbc_against_dbscan_avg_num_jumps(21)

    # compare_tcbc_against_dbscan_num_clusters(21)

    # compare_tcbc_against_dbscan_silhouette(21)

    # show_tcbc_experiment_metrics(historical=True, num_sources=21)
