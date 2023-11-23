import numpy as np
from matplotlib import pyplot as plt


def show_instant_epsilons(epsilon_values):
    epsilon_instant_means = {k: [value['clusters']['intra_mean'] for instant, value in v["instant"].items()]
                             for k, v in epsilon_values.items()}

    for key, values in epsilon_instant_means.items():
        plt.plot(values, label=key)

    epsilon_instant_means = {k: [value['clusters']['inter_mean'] for instant, value in v["instant"].items()]
                             for k, v in epsilon_values.items()}

    for key, values in epsilon_instant_means.items():
        plt.plot(values, label=key)

    plt.legend()
    plt.show()


def show_intra_mean_epsilons(epsilon_values):
    # Show plots
    epsilon_means = {k: v['intra_mean'] for k, v in epsilon_values.items()}

    for i, (key, value) in enumerate(epsilon_means.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.show()


def show_inter_mean_epsilons(epsilon_values):
    # Show plots
    epsilon_means = {k: v['inter_mean'] for k, v in epsilon_values.items()}

    for i, (key, value) in enumerate(epsilon_means.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.show()


def show_all_means_epsilons(epsilon_values):
    epsilon_means = {k: v['intra_mean'] for k, v in epsilon_values.items()}

    x = list(range(len(epsilon_means)))
    y = list(epsilon_means.values())
    plt.plot(x, y, label='intra_mean')  # Plot a line for intra_mean

    epsilon_means = {k: v['inter_mean'] for k, v in epsilon_values.items()}

    y = list(epsilon_means.values())
    plt.plot(x, y, label='inter_mean')  # Plot a line for inter_mean

    plt.legend()
    plt.show()


def show_all_means_silhouette(epsilon_values):
    epsilon_means = {k: v['intra_mean'] for k, v in epsilon_values.items()}

    x = list(range(len(epsilon_means)))
    y = list(epsilon_means.values())
    plt.plot(x, y, label='intra_mean')  # Plot a line for intra_mean

    epsilon_means = {k: v['silhouette_score'] for k, v in epsilon_values.items()}

    y = list(epsilon_means.values())
    plt.plot(x, y, label='silhouette_score')  # Plot a line for silhouette_score

    plt.legend()
    plt.show()


def show_all_means_instant(epsilon_values, epsilons, instant):
    epsilon_means = {k: v['instant'][instant]['clusters']['intra_mean'] for k, v in epsilon_values.items() if
                     k in epsilons}

    x = list(range(len(epsilon_means)))
    y = list(epsilon_means.values())
    plt.plot(x, y, label='intra_mean')  # Plot a line for intra_mean

    epsilon_means = {k: v['instant'][instant]['clusters']['inter_mean'] for k, v in epsilon_values.items() if
                     k in epsilons}

    y = list(epsilon_means.values())
    plt.plot(x, y, label='inter_mean')  # Plot a line for inter_mean

    plt.legend()
    plt.show()


def show_num_clusters_instant(epsilon_values, epsilons, instant):
    epsilon_clusters = {k: len(list(v['instant'][instant]['clusters']['value'].keys())) for k, v in epsilon_values.items() if
                     k in epsilons}

    epsilon_means = {k: v['instant'][instant]['clusters']['intra_mean'] for k, v in epsilon_values.items() if
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


def show_mean_num_clusters_epsilons(epsilon_values):
    # Show plots
    epsilon_clusters = {k: np.mean([len(value['clusters']['value'].keys()) for instant, value in v["instant"].items()])
                        for k, v in epsilon_values.items()}

    for i, (key, value) in enumerate(epsilon_clusters.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.show()

