import string

from temporal_series_clustering.cluster.algorithms.dbscan import dbscan_cluster
from temporal_series_clustering.cluster.algorithms.mean_cycle import mean_cycle_clustering
from temporal_series_clustering.cluster.clustering import temporal_clustering
from temporal_series_clustering.cluster.utils import store_clusters_json
from temporal_series_clustering.patterns.generators import create_simulation_weeks, predictor_city
from temporal_series_clustering.sheaf.sheaf_model import create_sheaf_model, create_simplified_sheaf_model
from temporal_series_clustering.sheaf.utils import propagate_sheaf_values
from temporal_series_clustering.static.constants import NUM_WEEKS, ITEMS_PER_DAY
from temporal_series_clustering.static.utils import process_clusters


def clusterize_real_predictors():
    num_vertices = 6
    sheaf_model = create_sheaf_model(num_vertices)
    # Get base vertices
    base_vertices = list(string.ascii_uppercase)[:num_vertices]

    # show_network(sheaf_model)

    num_weeks = NUM_WEEKS

    # Get the simulations of each predictor
    a_simulation = create_simulation_weeks(predictor_city, place_id='a', num_weeks=num_weeks)
    b_simulation = create_simulation_weeks(predictor_city, place_id='b', num_weeks=num_weeks)
    c_simulation = create_simulation_weeks(predictor_city, place_id='c', num_weeks=num_weeks)
    d_simulation = create_simulation_weeks(predictor_city, place_id='d', num_weeks=num_weeks)
    e_simulation = create_simulation_weeks(predictor_city, place_id='e', num_weeks=num_weeks)
    f_simulation = create_simulation_weeks(predictor_city, place_id='f', num_weeks=num_weeks)

    predictors_output = [a_simulation, b_simulation, c_simulation, d_simulation, e_simulation, f_simulation]

    # For testing purposes we are going to generate the output for each predictor
    reading_freq = 1

    summary, filtration, mean, values = propagate_sheaf_values(sheaf=sheaf_model,
                                                               predictors_output=predictors_output,
                                                               reading_freq=reading_freq,
                                                               num_items=NUM_WEEKS * ITEMS_PER_DAY * 7,
                                                               base_vertices=base_vertices)

    # Create instant sheaf simplified models
    instant_sheaf_models = {instant: create_simplified_sheaf_model(num_vertices, filtration[instant]) for instant in
                            range(len(filtration))}

    mean_edge_clusters = temporal_clustering(instant_sheaf_models, filtration, base_vertices,
                                             algorithm=mean_cycle_clustering,
                                             epsilon=0.07)

    clusters_data = process_clusters(clusters=mean_edge_clusters, predictions=predictors_output,
                                     consistencies=filtration)

    store_clusters_json(clusters_data, '../clusters/mean_edge.json')

    # Try the clustering with DBSCAN
    dbscan_clusters = temporal_clustering(instant_sheaf_models, filtration, base_vertices, algorithm=dbscan_cluster,
                                          epsilon=0.07)

    clusters_data = process_clusters(clusters=dbscan_clusters, predictions=predictors_output,
                                     consistencies=filtration)

    store_clusters_json(clusters_data, '../clusters/dbscan.json')


if __name__ == "__main__":
    print("CLUSTERING OF REAL DATA SOURCES:")
    clusterize_real_predictors()
    print()
