import pandas as pd
from matplotlib import pyplot as plt

from temporal_series_clustering.cluster.result_utils import load_output_clusters_file, get_avg_jumps, inverse_df_info, \
    get_jumps_epsilon, calculate_cost_function
from temporal_series_clustering.storage.epsilons import EpsilonValues

HEIGHT, WIDTH = 12, 8


def show_instants_intra_mean(epsilon_values: EpsilonValues):
    """
    Show a plot of intra_mean attribute for a given experiment output

    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :return:
    """
    plt.subplots(figsize=(HEIGHT, WIDTH))
    epsilon_instant_means = epsilon_values.get_instants_intra_mean()

    for key, values in epsilon_instant_means.items():
        plt.plot(values, label=key)

    epsilon_instant_means = epsilon_values.get_instants_inter_mean()

    for key, values in epsilon_instant_means.items():
        plt.plot(values, label=key)

    plt.legend()
    plt.show()


def show_avg_intra_mean(epsilon_values: EpsilonValues):
    """
    Show a plot of average intra_mean attribute for a given experiment output

    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :return:
    """
    plt.subplots(figsize=(HEIGHT, WIDTH))
    # Show plots
    epsilon_means = epsilon_values.get_avg_intra_mean()

    for i, (key, value) in enumerate(epsilon_means.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.show()


def show_avg_inter_mean(epsilon_values: EpsilonValues):
    """
    Show a plot of average inter_mean attribute for a given experiment output

    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :return:
    """
    plt.subplots(figsize=(HEIGHT, WIDTH))
    # Show plots
    epsilon_means = epsilon_values.get_avg_inter_mean()

    for i, (key, value) in enumerate(epsilon_means.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.show()


def compare_avg_intra_inter_means(epsilon_values: EpsilonValues):
    """
    Show a plot of average intra_mean and inter_mean attributes for a given experiment output

    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :return:
    """
    plt.subplots(figsize=(HEIGHT, WIDTH))
    epsilon_means = epsilon_values.get_avg_intra_mean()

    x = list(range(len(epsilon_means)))
    y = list(epsilon_means.values())
    plt.plot(x, y, label='intra_mean')  # Plot a line for intra_mean

    epsilon_means = epsilon_values.get_avg_inter_mean()

    y = list(epsilon_means.values())
    plt.plot(x, y, label='inter_mean')  # Plot a line for inter_mean

    plt.legend()
    plt.show()


def compare_intra_mean_silhouette(epsilon_values: EpsilonValues):
    """
    Show a plot of average intra_mean and silhouette score attributes for a given experiment output

    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :return:
    """
    plt.subplots(figsize=(HEIGHT, WIDTH))
    epsilon_means = epsilon_values.get_avg_intra_mean()

    x = list(range(len(epsilon_means)))
    y = list(epsilon_means.values())
    plt.plot(x, y, label='intra_mean')  # Plot a line for intra_mean

    epsilon_means = epsilon_values.get_avg_silhouette_score()
    y = list(epsilon_means.values())
    plt.plot(x, y, label='silhouette_score')  # Plot a line for silhouette_score

    plt.legend()
    plt.show()


def show_avg_num_clusters(epsilon_values: EpsilonValues):
    """
    Show a plot of average number of clusters for a given experiment output

    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :return:
    """
    plt.subplots(figsize=(HEIGHT, WIDTH))
    # Show plots
    epsilon_clusters = epsilon_values.get_avg_num_clusters()

    for i, (key, value) in enumerate(epsilon_clusters.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.show()


def show_avg_all_instants_metric_against_epsilon(metric: str, epsilon_values: EpsilonValues, results_dir: str,
                                                 show_plots: bool = False):
    """
    Show a plot of metric for a given experiment output. Can save the plot

    :param metric: metric to show
    :type metric: str
    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :param results_dir: results directory to store the plot
    :type results_dir: str
    :param show_plots: flag to show plot. Default to False
    :type show_plots: bool
    :return:
    """
    plt.subplots(figsize=(HEIGHT, WIDTH))
    data = {k: v[metric] for k, v in epsilon_values.info.items()}

    for i, (key, value) in enumerate(data.items()):
        plt.plot(i, value, 'o', label=key)
        plt.annotate(key, (i, value), xytext=(-10, 10), textcoords='offset points')

    plt.legend()
    plt.title(f"Average all instants against epsilon for metric {metric}")
    plt.savefig(results_dir + f"avg_all_instants_{metric}_against_epsilon.png")

    if show_plots:
        plt.show()


def show_metric_against_time(metric, epsilon_values: EpsilonValues, results_dir: str, show_plots: bool = False):
    """
    Show a plot of metric against time for a given experiment output. Can save the plot

    :param metric: metric to show
    :type metric: str
    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :param results_dir: results directory to store the plot
    :type results_dir: str
    :param show_plots: flag to show plot. Default to False
    :type show_plots: bool
    :return:
    """

    metric_data = {k: [value[metric] for instant, value in v["instant"].items()]
                   for k, v in epsilon_values.info.items()}

    num_clusters = epsilon_values.get_num_clusters()

    x = list(range(len(list(metric_data.values())[0])))
    values_num_clusters = list(num_clusters.values())
    values_metric_data = list(metric_data.values())

    fig, ax1 = plt.subplots(figsize=(HEIGHT, WIDTH))

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


def show_metric_against_time_by_epsilon(metric: str, epsilon: float, epsilon_values: EpsilonValues, results_dir: str,
                                        show_plots: bool = False):
    """
    Show a plot of metric against time for a given epsilon on experiment output. Can save the plot

    :param metric: metric to show
    :type metric: str
    :param epsilon: epsilon value
    :type epsilon: float
    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :param results_dir: results directory to store the plot
    :type results_dir: str
    :param show_plots: flag to show plot. Default to False
    :type show_plots: bool
    :return:
    """
    metric_data = {k: [value[metric] for instant, value in v["instant"].items()]
                   for k, v in epsilon_values.info.items() if k == epsilon}

    num_clusters = {k: v for k, v in epsilon_values.get_num_clusters().items() if k == epsilon}

    x = list(range(len(list(metric_data.values())[0])))
    values_num_clusters = list(num_clusters.values())
    values_metric_data = list(metric_data.values())

    fig, ax1 = plt.subplots(figsize=(HEIGHT, WIDTH))

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


def show_avg_num_clusters_against_epsilon(epsilon_values: EpsilonValues, results_dir: str, show_plots: bool = False):
    """
    Show a plot of average number of clusters against epsilon for a given experiment output. Can save the plot

    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :param results_dir: results directory to store the plot
    :type results_dir: str
    :param show_plots: flag to show plot. Default to False
    :type show_plots: bool
    :return:
    """
    plt.subplots(figsize=(HEIGHT, WIDTH))

    data = epsilon_values.get_num_clusters()

    for i, (key, value) in enumerate(data.items()):
        # [0] as it is a list
        plt.plot(i, value[0], 'o', label=key)
        plt.annotate(key, (i, value[0]), xytext=(-10, 10), textcoords='offset points')

    plt.legend()
    plt.title(f"Average number of clusters against epsilon")
    plt.savefig(results_dir + f"avg_num_clusters_against_epsilon.png")

    if show_plots:
        plt.show()


def show_num_clusters_against_time(epsilon_values: EpsilonValues, results_dir: str, show_plots: bool = False):
    """
    Show a plot of average number of clusters against epsilon for a given experiment output. Can save the plot

    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :param results_dir: results directory to store the plot
    :type results_dir: str
    :param show_plots: flag to show plot. Default to False
    :type show_plots: bool
    :return:
    """

    plt.subplots(figsize=(HEIGHT, WIDTH))

    data = epsilon_values.get_num_clusters()

    for key, values in data.items():
        plt.plot(values, label=key)

    plt.legend()
    plt.title(f"Number of clusters per epsilon against time")
    plt.savefig(results_dir + f"num_clusters_per_epsilon_against_time.png")

    if show_plots:
        plt.show()


def show_global_metric_against_time_by_epsilon(metric: str, epsilon: float, epsilon_values: EpsilonValues,
                                               results_dir: str, show_plots: bool = False):
    """
    Show a plot of metric against time for a given epsilon on experiment output. Can save the plot

    :param metric: metric to show
    :type metric: str
    :param epsilon: epsilon value
    :type epsilon: float
    :param epsilon_values: experiment output
    :type epsilon_values: EpsilonValues
    :param results_dir: results directory to store the plot
    :type results_dir: str
    :param show_plots: flag to show plot. Default to False
    :type show_plots: bool
    :return:
    """
    plt.subplots(figsize=(HEIGHT, WIDTH))

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


def compare_tcbc_avg_num_jumps(num_sources: int, inverse: bool = False):
    """
    Show a plot of average number of jumps comparing both historical and non-historical clustering

    :param num_sources: number of sources of experiment
    :type num_sources: int
    :param inverse: flag for enabling the inverse value
    :type inverse: bool
    :return:
    """
    values_historical = load_output_clusters_file(f'../results/experiment_{num_sources}_sources_historical.json')
    values_no_historical = load_output_clusters_file(f'../results/experiment_{num_sources}_sources_no_historical.json')

    # Create a dataframe
    df_historical_info, df_no_historical_info = pd.DataFrame(), pd.DataFrame()
    # Append average number of jumps
    df_historical_info['avg_jumps'] = get_avg_jumps(values_historical)
    # Append average number of jumps
    df_no_historical_info['avg_jumps'] = get_avg_jumps(values_no_historical)

    if inverse:
        df_historical_info = inverse_df_info(df_historical_info)
        df_no_historical_info = inverse_df_info(df_no_historical_info)

    # Create the plot
    plt.figure(figsize=(HEIGHT, WIDTH))
    plt.plot(df_historical_info['avg_jumps'], label="historical")
    plt.plot(df_no_historical_info['avg_jumps'], label="no_historical")
    plt.ylabel('Mean of jumps')
    plt.xlabel('Epsilon')
    plt.title('Mean of jumps for all instants for each epsilon')
    plt.legend()
    plt.show()


def compare_tcbc_num_jumps_epsilon(num_sources: int, epsilon: str):
    """
    Show a plot of number of jumps comparing both historical and non-historical clustering

    :param num_sources: number of sources of experiment
    :type num_sources: int
    :param epsilon: epsilon value
    :type epsilon: str
    :return:
    """
    values_historical = load_output_clusters_file(f'../results/experiment_{num_sources}_sources_historical.json')
    values_no_historical = load_output_clusters_file(
        f'../results/experiment_{num_sources}_sources_no_historical.json')

    # Create a dataframe with the number of jumps on each approach
    df_historical_info = pd.DataFrame(get_jumps_epsilon(values_historical, epsilon))
    df_no_historical_info = pd.DataFrame(get_jumps_epsilon(values_no_historical, epsilon))

    # Create the plot
    plt.figure(figsize=(HEIGHT, WIDTH))
    plt.plot(df_historical_info.index, df_historical_info.values, label="historical")
    plt.plot(df_no_historical_info.index, df_no_historical_info.values, label="no_historical")
    plt.ylabel('Number of jumps')
    plt.xlabel('Instant')
    plt.title('Number of jumps for all instants for each instant')
    plt.legend()
    plt.show()


def compare_tcbc_num_clusters_coherence(num_sources: int):
    """
    Show a plot comparing inverse of average number of jumps (y1-axis) and average number of clusters (y2-axis)

    :param num_sources: number of sources of experiment
    :type num_sources: int
    :return:
    """
    values_historical = load_output_clusters_file(f'../results/experiment_{num_sources}_sources_historical.json')
    values_no_historical = load_output_clusters_file(
        f'../results/experiment_{num_sources}_sources_no_historical.json')

    # Create a dataframe
    df_historical_info, df_no_historical_info = pd.DataFrame(), pd.DataFrame()
    # Append average number of jumps
    df_historical_info['avg_jumps'] = get_avg_jumps(values_historical)
    df_no_historical_info['avg_jumps'] = get_avg_jumps(values_no_historical)

    # Append average number of clusters
    df_historical_info['avg_num_clusters'] = values_historical.get_avg_num_clusters()
    df_no_historical_info['avg_num_clusters'] = values_no_historical.get_avg_num_clusters()

    fig, ax1 = plt.subplots(figsize=(HEIGHT, WIDTH))

    # Plotting the 'avg_jumps' from the three dataframes on the first y-axis
    ax1.plot(1 / df_historical_info['avg_jumps'], color='red', label='tcbc_historical')
    ax1.plot(1 / df_no_historical_info['avg_jumps'], color='green', label='tcbc_no_historical')
    ax1.set_ylabel('avg_jumps', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Creating the second axis
    ax2 = ax1.twinx()

    # Plotting the 'avg_num_clusters' from the three dataframes on the second y-axis
    ax2.plot(df_historical_info['avg_num_clusters'], color='red', linestyle='dashed')
    ax2.plot(df_no_historical_info['avg_num_clusters'], color='green', linestyle='dashed')
    ax2.set_ylabel('avg_num_clusters', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Adding a legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()


def plot_cost_function_tcbc(num_sources: int):
    """
    Plot the cost function for the TCBC algorithm.

    f = std(avg_jumps) - std(avg_silhouette_score)

    :param num_sources: number of sources of experiment
    :type num_sources: int
    :return:
    """
    tcbc_no_historical_jumps_df = calculate_cost_function(
        f'../results/experiment_{num_sources}_sources_no_historical.json')
    tcbc_historical_jumps_df = calculate_cost_function(
        f'../results/experiment_{num_sources}_sources_historical.json')

    # Crear la gráfica
    plt.figure(figsize=(HEIGHT, WIDTH))
    plt.plot(tcbc_historical_jumps_df['f'], label="tcbc_historical")
    plt.plot(tcbc_no_historical_jumps_df['f'], label="tcbc_no_historical")
    plt.ylabel('Cost function')
    plt.xlabel('Epsilon')
    plt.title('Cost function for each epsilon')
    plt.legend()
    plt.show()


def compare_tcbc_against_dbscan_avg_num_jumps(num_sources: int):
    """
    Show a plot of average number of jumps comparing TCBC historical and non-historical and DBSCAN clustering


    :param num_sources: number of sources of experiment
    :type num_sources: int
    :return:
    """
    # DBSCAN
    dbscan_values = load_output_clusters_file(f'../results/dbscan_{num_sources}_sources.json')

    # Create a dataframe
    df_dbscan_values = pd.DataFrame()
    # Append average number of jumps parsing key to float
    avg_jumps = {float(k): v for k, v in get_avg_jumps(dbscan_values).items()}
    # On DBScan, append the epsilon 0.0
    avg_jumps[0.00] = 0.0
    # Sort the dataframe, so the 0.00 is in first place
    df_dbscan_values['avg_jumps'] = dict(sorted(avg_jumps.items()))
    # Change index to str
    df_dbscan_values.index = df_dbscan_values.index.astype(str)

    values_historical = load_output_clusters_file(f'../results/experiment_{num_sources}_sources_historical.json')
    values_no_historical = load_output_clusters_file(f'../results/experiment_{num_sources}_sources_no_historical.json')

    # Create a dataframe
    df_historical_info, df_no_historical_info = pd.DataFrame(), pd.DataFrame()
    # Append average number of jumps
    df_historical_info['avg_jumps'] = get_avg_jumps(values_historical)
    # Append average number of jumps
    df_no_historical_info['avg_jumps'] = get_avg_jumps(values_no_historical)

    # Create the plot
    plt.figure(figsize=(HEIGHT, WIDTH))
    plt.plot(df_historical_info['avg_jumps'], label="tcbc_historical")
    plt.plot(df_no_historical_info['avg_jumps'], label="tcbc_no_historical")
    plt.plot(df_dbscan_values['avg_jumps'], label="DBSCAN")
    plt.ylabel('Mean of jumps')
    plt.xlabel('Epsilon')
    plt.title('Mean of jumps for all instants for each epsilon')
    plt.legend()
    plt.show()


def compare_tcbc_against_dbscan_num_clusters(num_sources: int):
    """
    Show a plot of number of clusters comparing TCBC historical and non-historical and DBSCAN clustering


    :param num_sources: number of sources of experiment
    :type num_sources: int
    :return:
    """
    # DBSCAN
    dbscan_values = load_output_clusters_file(f'../results/dbscan_{num_sources}_sources.json')

    # Create a dataframe
    df_dbscan_values = pd.DataFrame()
    # Append average number of jumps parsing key to float
    avg_jumps = {float(k): v for k, v in get_avg_jumps(dbscan_values).items()}
    # On DBScan, append the epsilon 0.0
    avg_jumps[0.00] = 0.0
    # Sort the dataframe, so the 0.00 is in first place
    df_dbscan_values['avg_jumps'] = dict(sorted(avg_jumps.items()))
    # Change index to str
    df_dbscan_values.index = df_dbscan_values.index.astype(str)

    values_historical = load_output_clusters_file(f'../results/experiment_{num_sources}_sources_historical.json')
    values_no_historical = load_output_clusters_file(f'../results/experiment_{num_sources}_sources_no_historical.json')

    # Create a dataframe
    df_historical_info, df_no_historical_info = pd.DataFrame(), pd.DataFrame()
    # Append average number of jumps
    df_historical_info['avg_jumps'] = get_avg_jumps(values_historical)
    # Append average number of jumps
    df_no_historical_info['avg_jumps'] = get_avg_jumps(values_no_historical)

    # Append average number of clusters
    df_dbscan_values['avg_num_clusters'] = dbscan_values.get_avg_num_clusters()
    # Replace NaN to 0
    df_dbscan_values.fillna(0, inplace=True)
    df_historical_info['avg_num_clusters'] = values_historical.get_avg_num_clusters()
    df_no_historical_info['avg_num_clusters'] = values_no_historical.get_avg_num_clusters()

    fig, ax1 = plt.subplots(figsize=(HEIGHT, WIDTH))

    # Plotting the 'avg_jumps' from the three dataframes on the first y-axis
    ax1.plot(df_dbscan_values['avg_jumps'], color='blue', label='DBSCAN')
    ax1.plot(df_historical_info['avg_jumps'], color='red', label='tcbc_historical')
    ax1.plot(df_no_historical_info['avg_jumps'], color='green', label='tcbc_no_historical')
    ax1.set_ylabel('avg_jumps', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Creating the second axis
    ax2 = ax1.twinx()

    # Plotting the 'avg_num_clusters' from the three dataframes on the second y-axis
    ax2.plot(df_dbscan_values['avg_num_clusters'], color='blue', linestyle='dashed')
    ax2.plot(df_historical_info['avg_num_clusters'], color='red', linestyle='dashed')
    ax2.plot(df_no_historical_info['avg_num_clusters'], color='green', linestyle='dashed')
    ax2.set_ylabel('avg_num_clusters', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Adding a legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()


def compare_tcbc_against_dbscan_silhouette(num_sources: int):
    """
    Show a plot of silhouette score comparing TCBC historical and non-historical and DBSCAN clustering


    :param num_sources: number of sources of experiment
    :type num_sources: int
    :return:
    """
    # DBSCAN
    dbscan_values = load_output_clusters_file(f'../results/dbscan_{num_sources}_sources.json')

    # Create a dataframe
    df_dbscan_values = pd.DataFrame()
    # Append average number of jumps parsing key to float
    avg_jumps = {float(k): v for k, v in get_avg_jumps(dbscan_values).items()}
    # On DBScan, append the epsilon 0.0
    avg_jumps[0.00] = 0.0
    # Sort the dataframe, so the 0.00 is in first place
    df_dbscan_values['avg_jumps'] = dict(sorted(avg_jumps.items()))
    # Change index to str
    df_dbscan_values.index = df_dbscan_values.index.astype(str)

    values_historical = load_output_clusters_file(f'../results/experiment_{num_sources}_sources_historical.json')
    values_no_historical = load_output_clusters_file(f'../results/experiment_{num_sources}_sources_no_historical.json')

    # Create a dataframe
    df_historical_info, df_no_historical_info = pd.DataFrame(), pd.DataFrame()
    # Append average number of jumps
    df_historical_info['avg_jumps'] = get_avg_jumps(values_historical)
    # Append average number of jumps
    df_no_historical_info['avg_jumps'] = get_avg_jumps(values_no_historical)

    # Append silhouette
    df_dbscan_values['silhouette_score'] = dbscan_values.get_avg_silhouette_score()
    # Replace NaN to 0
    df_dbscan_values.fillna(0, inplace=True)
    df_historical_info['silhouette_score'] = values_historical.get_avg_silhouette_score()
    df_no_historical_info['silhouette_score'] = values_no_historical.get_avg_silhouette_score()

    fig, ax1 = plt.subplots(figsize=(HEIGHT, WIDTH))

    # Plotting the 'avg_jumps' from the three dataframes on the first y-axis
    ax1.plot(df_dbscan_values['avg_jumps'], color='blue', label='DBSCAN')
    ax1.plot(df_historical_info['avg_jumps'], color='red', label='tcbc_historical')
    ax1.plot(df_no_historical_info['avg_jumps'], color='green', label='tcbc_no_historical')
    ax1.set_ylabel('avg_jumps', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Creating the second axis
    ax2 = ax1.twinx()

    # Plotting the 'silhouette_score' from the three dataframes on the second y-axis
    ax2.plot(df_dbscan_values['silhouette_score'], color='blue', linestyle='dashed')
    ax2.plot(df_historical_info['silhouette_score'], color='red', linestyle='dashed')
    ax2.plot(df_no_historical_info['silhouette_score'], color='green', linestyle='dashed')
    ax2.set_ylabel('silhouette_score', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Adding a legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()


def show_kmeans_avg_jumps(num_sources: int):
    """
    Show the average number of jumps per k using the kmeans algorithm

    :param num_sources: number of sources of the experiment
    :type num_sources: int
    :return:
    """
    values = load_output_clusters_file(f'../results/kmeans_{num_sources}_sources.json')

    # Create a dataframe
    df_values = pd.DataFrame()
    # Append average number of jumps
    df_values['avg_jumps'] = get_avg_jumps(values)

    # Create the plot
    plt.figure(figsize=(HEIGHT, WIDTH))
    plt.plot(df_values['avg_jumps'], label="KMEANS")
    plt.ylabel('Mean of jumps')
    plt.xlabel('K')
    plt.title('Mean of jumps for all instants for each K (KMEANS)')
    plt.legend()
    plt.show()


def show_dbscan_avg_jumps(num_sources: int):
    """
    Show the average number of jumps per epsilon using the DBSCAN algorithm

    :param num_sources: number of sources of the experiment
    :type num_sources: int
    :return:
    """
    values = load_output_clusters_file(f'../results/dbscan_{num_sources}_sources.json')

    # Create a dataframe
    df_values = pd.DataFrame()
    # Append average number of jumps parsing key to float
    avg_jumps = {float(k): v for k, v in get_avg_jumps(values).items()}
    # On DBScan, append the epsilon 0.0
    avg_jumps[0.00] = 0.0
    # Sort the dataframe, so the 0.00 is in first place
    df_values['avg_jumps'] = dict(sorted(avg_jumps.items()))

    # Create the plot
    plt.figure(figsize=(HEIGHT, WIDTH))
    plt.plot(df_values['avg_jumps'], label="DBSCAN")
    plt.ylabel('Mean of jumps')
    plt.xlabel('Epsilon')
    plt.title('Mean of jumps for all instants for each epsilon (DBSCAN)')
    plt.legend()
    plt.show()
