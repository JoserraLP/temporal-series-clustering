import numpy as np
from matplotlib import pyplot as plt

from temporal_series_clustering.storage.epsilons import EpsilonValues

results_dir = "../results/plots/"


def show_instant_epsilons(epsilon_values: EpsilonValues):
    epsilon_instant_means = {k: [value['intra_mean'] for instant, value in v["instant"].items()]
                             for k, v in epsilon_values.info.items()}

    for key, values in epsilon_instant_means.items():
        plt.plot(values, label=key)

    epsilon_instant_means = {k: [value['inter_mean'] for instant, value in v["instant"].items()]
                             for k, v in epsilon_values.info.items()}

    for key, values in epsilon_instant_means.items():
        plt.plot(values, label=key)

    plt.legend()
    plt.show()


def show_intra_mean_epsilons(epsilon_values: EpsilonValues):
    # Show plots
    epsilon_means = {k: v['intra_mean'] for k, v in epsilon_values.info.items()}

    for i, (key, value) in enumerate(epsilon_means.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.show()


def show_inter_mean_epsilons(epsilon_values):
    # Show plots
    epsilon_means = {k: v['inter_mean'] for k, v in epsilon_values.info.items()}

    for i, (key, value) in enumerate(epsilon_means.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.show()


def show_all_means_epsilons(epsilon_values: EpsilonValues):
    epsilon_means = {k: v['intra_mean'] for k, v in epsilon_values.info.items()}

    x = list(range(len(epsilon_means)))
    y = list(epsilon_means.values())
    plt.plot(x, y, label='intra_mean')  # Plot a line for intra_mean

    epsilon_means = {k: v['inter_mean'] for k, v in epsilon_values.info.items()}

    y = list(epsilon_means.values())
    plt.plot(x, y, label='inter_mean')  # Plot a line for inter_mean

    plt.legend()
    plt.show()


def show_all_means_silhouette(epsilon_values: EpsilonValues):
    epsilon_means = {k: v['intra_mean'] for k, v in epsilon_values.info.items()}

    x = list(range(len(epsilon_means)))
    y = list(epsilon_means.values())
    plt.plot(x, y, label='intra_mean')  # Plot a line for intra_mean

    epsilon_means = {k: v['silhouette_score'] for k, v in epsilon_values.info.items()}

    y = list(epsilon_means.values())
    plt.plot(x, y, label='silhouette_score')  # Plot a line for silhouette_score

    plt.legend()
    plt.show()


def show_all_means_instant(epsilon_values, epsilons: EpsilonValues, instant):
    epsilon_means = {k: v['instant'][instant]['intra_mean'] for k, v in epsilon_values.info.items() if
                     k in epsilons}

    x = list(range(len(epsilon_means)))
    y = list(epsilon_means.values())
    plt.plot(x, y, label='intra_mean')  # Plot a line for intra_mean

    epsilon_means = {k: v['instant'][instant]['inter_mean'] for k, v in epsilon_values.info.items() if
                     k in epsilons}

    y = list(epsilon_means.values())
    plt.plot(x, y, label='inter_mean')  # Plot a line for inter_mean

    plt.legend()
    plt.show()


def show_num_clusters_instant(epsilon_values, epsilons: EpsilonValues, instant):
    epsilon_clusters = {k: len(list(v['instant'][instant]['value'].keys())) for k, v in epsilon_values.info.items() if
                        k in epsilons}

    epsilon_means = {k: v['instant'][instant]['value'] for k, v in epsilon_values.info.items() if
                     k in epsilons}

    x = list(range(len(epsilon_clusters)))
    num_clusters = list(epsilon_clusters.values())
    intra_mean = list(epsilon_means.values())

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('epsilon')
    ax1.set_ylabel('num_clusters', color=color)
    ax1.plot(x, num_clusters, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('intra-mean', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, intra_mean, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def show_mean_num_clusters_epsilons(epsilon_values: EpsilonValues):
    # Show plots
    epsilon_clusters = {k: np.mean([len(value['value'].keys()) for instant, value in v["instant"].items()])
                        for k, v in epsilon_values.info.items()}

    for i, (key, value) in enumerate(epsilon_clusters.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.show()


### From here on are the plots defined by Victor and Me
def show_avg_all_instants_metric_against_epsilon(metric, epsilon_values: EpsilonValues, show_plots: bool = False):
    data = {k: v[metric] for k, v in epsilon_values.info.items()}

    for i, (key, value) in enumerate(data.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.legend()
    plt.title(f"Average all instants against epsilon for metric {metric}")
    plt.savefig(results_dir + f"avg_all_instants_{metric}_against_epsilon.png")

    if show_plots:
        plt.show()
    else:
        plt.clf()


def show_metric_against_time(metric, epsilon_values: EpsilonValues, show_plots: bool = False):
    metric_data = {k: [value[metric] for instant, value in v["instant"].items()]
                   for k, v in epsilon_values.info.items()}

    num_clusters = {k: [len(list(v['instant'][instant]['value'].keys())) for instant, value in v["instant"].items()]
                    for k, v in epsilon_values.info.items()}
    x = list(range(len(list(metric_data.values())[0])))
    values_num_clusters = list(num_clusters.values())
    values_metric_data = list(metric_data.values())

    fig, ax1 = plt.subplots()

    # Create a colormap
    cmap = plt.get_cmap('hsv')

    for i, epsilon in enumerate(list(metric_data.keys())):
        # Map the index to a color
        color = cmap(i / len(values_metric_data))

        ax1.plot(x, values_metric_data[i], color=color, label=epsilon, linestyle='-')

    ax1.set_xlabel('time')
    ax1.set_ylabel(metric)

    ax2 = ax1.twinx()

    for i, epsilon in enumerate(list(num_clusters.keys())):
        # Map the index to a color
        color = cmap(i / len(values_num_clusters))

        ax2.plot(x, values_num_clusters[i], color=color, label=epsilon, linestyle='--')

    ax2.set_ylabel('num_clusters')  # we already handled the x-label with ax1

    # Add text below the plot
    plt.figtext(0.5, 0.01, f"Straight lines:{metric};\nDashed lines:num_clusters",
                ha="center", fontsize=10)
    plt.legend()
    plt.title(f"{metric} against time compared to number of clusters")
    plt.savefig(results_dir + f"{metric}_and_num_clusters_against_time.png")

    if show_plots:
        plt.show()
    else:
        plt.clf()


def show_metric_against_time_by_epsilon(metric, epsilon, epsilon_values: EpsilonValues, show_plots: bool = False):
    metric_data = {k: [value[metric] for instant, value in v["instant"].items()]
                   for k, v in epsilon_values.info.items() if k == epsilon}

    num_clusters = {k: [len(list(v['instant'][instant]['value'].keys())) for instant, value in v["instant"].items()]
                    for k, v in epsilon_values.info.items() if k == epsilon}
    x = list(range(len(list(metric_data.values())[0])))
    values_num_clusters = list(num_clusters.values())
    values_metric_data = list(metric_data.values())

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.plot(x, values_metric_data[0], color=color, label=epsilon, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('time')
    ax1.set_ylabel(metric, color=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.plot(x, values_num_clusters[0], color=color, label=epsilon, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('num_clusters', color=color)  # we already handled the x-label with ax1

    # Add text below the plot
    plt.figtext(0.5, 0.01, f"Straight lines:{metric};\nDashed lines:num_clusters",
                ha="center", fontsize=10)
    plt.legend()
    plt.title(f"{metric} against time compared to number of clusters for epsilon {epsilon}")
    plt.savefig(results_dir + f"{metric}_and_num_clusters_for_epsilon_{epsilon}_against_time.png")

    if show_plots:
        plt.show()
    else:
        plt.clf()


def show_avg_num_clusters_against_epsilon(epsilon_values: EpsilonValues, show_plots: bool = False):
    data = {k: len(list(v['instant'][instant]['value'].keys())) for k, v in epsilon_values.info.items()
            for instant, value in v["instant"].items()}

    for i, (key, value) in enumerate(data.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.legend()
    plt.title(f"Average number of clusters against epsilon")
    plt.savefig(results_dir + f"avg_num_clusters_against_epsilon.png")

    if show_plots:
        plt.show()
    else:
        plt.clf()


def show_num_clusters_against_time(epsilon_values: EpsilonValues, show_plots: bool = False):
    data = {k: [len(list(v['instant'][instant]['value'].keys())) for instant, value in v["instant"].items()]
            for k, v in epsilon_values.info.items()}

    for key, values in data.items():
        plt.plot(values, label=key)

    plt.legend()
    plt.title(f"Number of clusters per epsilon against time")
    plt.savefig(results_dir + f"num_clusters_per_epsilon_against_time.png")

    if show_plots:
        plt.show()
    else:
        plt.clf()


def show_global_metric_against_time_by_epsilon(metric: str, epsilon: float, epsilon_values: EpsilonValues,
                                               show_plots: bool = False):
    data = {k: v[metric] for k, v in epsilon_values.info.items() if k == epsilon}

    if data:
        for i, (key, value) in enumerate(data.items()):
            plt.plot(i, value, 'o', label=key)
            plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

            plt.legend()
            plt.title(f"Global {metric} for epsilon {epsilon} against time")
            plt.savefig(results_dir + f"global_{metric}_for_epsilon_{epsilon}_against_time.png")

            if show_plots:
                plt.show()
            else:
                plt.clf()
