import random

import numpy as np

from temporal_series_clustering.static.constants import TEMPORAL_PATTERN_FILES, ITEMS_PER_DAY
from scipy import interpolate

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


def create_simulation_days(predictor, num_days):
    return [predictor(hour=hour) for hour in range(ITEMS_PER_DAY)] * num_days


def create_simulation_specific_weekday(predictor, place_id, weekday):
    simulation_normal_vol = [predictor(place_id=place_id, weekday=weekday, hour=hour, location="") for hour in
                             range(ITEMS_PER_DAY)]
    return simulation_normal_vol


def create_simulation_weeks(predictor, place_id, num_weeks):
    simulation_normal_vol = []
    for _ in range(num_weeks):
        simulation_normal_vol.extend(
            [predictor(place_id=place_id, weekday="weekday", hour=hour, location="") for hour in
             range(ITEMS_PER_DAY)] * 5)
        simulation_normal_vol.extend(
            [predictor(place_id=place_id, weekday="saturday", hour=hour, location="") for hour in
             range(ITEMS_PER_DAY)])
        simulation_normal_vol.extend([predictor(place_id=place_id, weekday="sunday", hour=hour, location="") for hour in
                                      range(ITEMS_PER_DAY)])

    return simulation_normal_vol


def interpolate_time_serie(time_serie: list, num_points: int) -> list:
    """
    Interpolate the input time serie, adding the number of items by parameters between each pair of values

    :param time_serie: time series values
    :type time_serie: list
    :param num_points: number of points between two pairs on the time serie
    :type num_points: int
    :return: interpolated time serie
    :rtype: list
    """

    # Create an array of times corresponding to values
    times = np.arange(len(time_serie))

    # Create a cubic spline interpolation function
    f = interpolate.interp1d(times, time_serie, kind='cubic')

    # Create a new array of times with the number of points between each pair of original points
    new_times = np.linspace(0, len(time_serie) - 1, num_points * (len(time_serie) - 1) + 1)

    # Use the interpolation function to get the interpolated values
    interpolated_values = f(new_times)

    return interpolated_values


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


def add_offset(arr, offset):
    offset = offset % len(arr)  # Handle offsets greater than the length
    return arr[-offset:] + arr[:-offset]


def add_instant_variation(arr, noise_level=0.1, seed=1):
    # Define the set
    random.seed(seed)
    # Limit the value between 0 and 1
    return [min(max(x + random.uniform(-noise_level, noise_level), 0), 1) for x in arr]


def combine_patterns(patterns, seed=1):
    # Get the length of the patterns
    length = len(patterns[0])

    random.seed(seed)

    # Check that all patterns are the same length
    if not all(len(pattern) == length for pattern in patterns):
        raise ValueError("All patterns must be the same length")

    # Create a new pattern with the same length
    new_pattern = []
    for i in range(length):
        # For each instant, choose a pattern randomly and take its value for that instant
        pattern = random.choice(patterns)
        new_pattern.append(pattern[i])

    return new_pattern


def smooth_edgy_values(pattern, threshold=0.40, smoothing_factor=0.40):
    smoothed = [pattern[0]]  # Start with the first value
    for i in range(1, len(pattern)):
        diff = pattern[i] - pattern[i - 1]
        if abs(diff) > threshold:
            # If the difference is too large, smooth the value
            smoothed_value = smoothed[-1] + diff * smoothing_factor
            smoothed.append(smoothed_value)
        else:
            # If the difference is not too large, keep the original value
            smoothed.append(pattern[i])
    return smoothed
