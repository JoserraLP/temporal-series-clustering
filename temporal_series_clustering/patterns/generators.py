import numpy as np

from temporal_series_clustering.static.constants import TEMPORAL_PATTERN_FILES, ITEMS_PER_DAY

import pandas as pd

TIME_PATTERN_A_B = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
TIME_PATTERN_C_D = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
TIME_PATTERN_E = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def time_pattern_a_b(hour: int) -> int:
    return TIME_PATTERN_A_B[hour]


def time_pattern_c_d(hour: int) -> int:
    return TIME_PATTERN_C_D[hour]


def time_pattern_e(hour: int) -> int:
    return TIME_PATTERN_E[hour]


def predictor_city(place_id: str, weekday: str, hour: int, location: str) -> int:
    place_file = TEMPORAL_PATTERN_FILES.get(place_id, "")

    df = pd.read_json(place_file)
    df_selection = df.filter(like=weekday, axis=1).copy()

    # Get name of the selected column
    df_selection_column = list(df_selection)[0]

    # Add hour value
    df_selection.loc[:, 'hour'] = df['hour'].values

    df_selection = df_selection.loc[df_selection['hour'] == hour]

    return df_selection[df_selection_column].values[0]


def add_noise(arr, range_value, min_value, max_value, seed=1):
    # Parse to numpy
    if type(arr) == list:
        arr = np.array(arr)
    # Define your noise range
    noise_range = (-range_value, range_value)

    np.random.seed(seed)

    # Generate random noise within the range
    if range_value == 1:
        # This is for congestion
        noise = np.random.randint(noise_range[0], noise_range[1], arr.shape)
    else:
        noise = np.random.uniform(noise_range[0], noise_range[1], arr.shape)

    # Add the noise to the array
    arr = arr + noise

    # Clip the array
    arr = np.clip(arr, min_value, max_value)

    return arr


def create_simulation_days(predictor, num_days):
    return [predictor(hour=hour) for hour in range(ITEMS_PER_DAY)] * num_days


def create_simulation_weeks(predictor, place_id, num_weeks):
    simulation_normal_vol = []
    for _ in range(num_weeks):
        simulation_normal_vol.extend([predictor(place_id=place_id, weekday="weekday", hour=hour, location="") for hour in
                                      range(ITEMS_PER_DAY)] * 5)
        simulation_normal_vol.extend([predictor(place_id=place_id, weekday="saturday", hour=hour, location="") for hour in
                                      range(ITEMS_PER_DAY)])
        simulation_normal_vol.extend([predictor(place_id=place_id, weekday="sunday", hour=hour, location="") for hour in
                                      range(ITEMS_PER_DAY)])

    return simulation_normal_vol

