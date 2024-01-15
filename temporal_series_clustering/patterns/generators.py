import random

import numpy as np

from temporal_series_clustering.static.constants import TEMPORAL_PATTERN_FILES, ITEMS_PER_DAY
from scipy import interpolate

import pandas as pd

TIME_PATTERN_A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
TIME_PATTERN_B = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
TIME_PATTERN_C = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

TIME_PATTERN_D = [0.00, 0.02, 0.00, 0.02, 0.00, 0.02, 0.00, 0.02, 0.00, 0.02, 0.00, 0.02, 0.00, 0.02, 0.00, 0.02,
                  0.00, 0.02, 0.00, 0.02, 0.00, 0.02, 0.00, 0.02]
TIME_PATTERN_E = [0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12,
                  0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12]
TIME_PATTERN_F = [0.05, 0.07, 0.05, 0.07, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.07, 0.09, 0.09, 0.09, 0.11,
                  0.11, 0.09, 0.07, 0.07, 0.05, 0.07, 0.05, 0.07]


def time_pattern_a(instant: int) -> int:
    """
    Get the value of time_pattern_a for a given instant

    :param instant: instant to get the value
    :type instant: int
    :return: instant value
    """
    return TIME_PATTERN_A[instant]


def time_pattern_b(instant: int) -> int:
    """
    Get the value of time_pattern_b for a given instant

    :param instant: instant to get the value
    :type instant: int
    :return: instant value
    """
    return TIME_PATTERN_B[instant]


def time_pattern_c(instant: int) -> int:
    """
    Get the value of time_pattern_c for a given instant

    :param instant: instant to get the value
    :type instant: int
    :return: instant value
    """
    return TIME_PATTERN_C[instant]


def time_pattern_d(instant: int) -> int:
    """
    Get the value of time_pattern_d for a given instant

    :param instant: instant to get the value
    :type instant: int
    :return: instant value
    """
    return TIME_PATTERN_D[instant]


def time_pattern_e(instant: int) -> int:
    """
    Get the value of time_pattern_e for a given instant

    :param instant: instant to get the value
    :type instant: int
    :return: instant value
    """
    return TIME_PATTERN_E[instant]


def time_pattern_f(instant: int) -> int:
    """
    Get the value of time_pattern_f for a given instant

    :param instant: instant to get the value
    :type instant: int
    :return: instant value
    """
    return TIME_PATTERN_F[instant]


def predictor_city(place_id: str, weekday: str, hour: int, location: str = ""):
    """
    Get the instant value for a given weekday and place

    :param place_id: data source place identifier
    :type place_id: str
    :param weekday: weekday
    :type weekday: str
    :param hour: hour selected
    :type hour: int
    :param location: location of the predictor
    :type location: str
    :return: value for the specific place, weekday and hour
    """
    # Get place file
    place_file = TEMPORAL_PATTERN_FILES.get(place_id, "")

    # Cerate a dataframe with the place file
    df = pd.read_json(place_file)
    # Filter by weekday
    df_selection = df.filter(like=weekday, axis=1).copy()

    # Get name of the selected column
    df_selection_column = list(df_selection)[0]

    # Add hour value
    df_selection.loc[:, 'hour'] = df['hour'].values

    # Get the value with same hour
    df_selection = df_selection.loc[df_selection['hour'] == hour]

    return df_selection[df_selection_column].values[0]


def create_simulation_days(predictor, num_days):
    """
    Create simulation for a given predictor and a given number of days

    :param predictor: time pattern predictor as method
    :param num_days: number of days to simulate
    :return: list with the time pattern
    """
    return [predictor(hour=hour) for hour in range(ITEMS_PER_DAY)] * num_days


def create_simulation_specific_weekday(predictor, place_id: str, weekday: str):
    """
    Create a simulation for a specific weekday and a given place

    :param predictor: time pattern predictor as method
    :param place_id: data source place identifier
    :type place_id: str
    :param weekday: weekday
    :type weekday: str
    :return: list with the time pattern
    """
    return [predictor(place_id=place_id, weekday=weekday, hour=hour, location="") for hour in
            range(ITEMS_PER_DAY)]


def create_simulation_weeks(predictor, place_id: str, num_weeks: int):
    """
    Create a simulation for a number of weeks and a given place

    :param predictor: time pattern predictor as method
    :param place_id: data source place identifier
    :type place_id: str
    :param num_weeks: number of weeks to simulate
    :type num_weeks: int
    :return: list with the time pattern
    """
    simulation_normal_vol = []
    # Iterate over the number of weeks
    for _ in range(num_weeks):
        # Append 5 weekdays
        simulation_normal_vol.extend(
            [predictor(place_id=place_id, weekday="weekday", hour=hour, location="") for hour in
             range(ITEMS_PER_DAY)] * 5)
        # Append a saturday
        simulation_normal_vol.extend(
            [predictor(place_id=place_id, weekday="saturday", hour=hour, location="") for hour in range(ITEMS_PER_DAY)])
        # Append a sunday
        simulation_normal_vol.extend(
            [predictor(place_id=place_id, weekday="sunday", hour=hour, location="") for hour in range(ITEMS_PER_DAY)])

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


def add_noise(time_pattern: list, range_value, min_value, max_value, seed=1):
    """
    Add specific range noise to a given time pattern, limited by a range of values

    :param time_pattern: input time pattern to add noise
    :type time_pattern: list
    :param range_value: range of noise
    :param min_value: minimum value allowed on resulting pattern
    :param max_value: maximum value allowed on resulting pattern
    :param seed: seed for random noise
    :return: resulting pattern with noise
    """
    # Parse to numpy
    if type(time_pattern) == list:
        time_pattern = np.array(time_pattern)
    # Define noise range
    noise_range = (-range_value, range_value)
    # Define seed
    np.random.seed(seed)

    # Generate random noise within the range
    if range_value == 1:
        # This is for congestion
        noise = np.random.randint(noise_range[0], noise_range[1], time_pattern.shape)
    else:
        noise = np.random.uniform(noise_range[0], noise_range[1], time_pattern.shape)

    # Add the noise to the array
    time_pattern = time_pattern + noise

    # Clip the array
    time_pattern = np.clip(time_pattern, min_value, max_value)

    return time_pattern


def add_offset(time_pattern: list, offset):
    """
    Add offset to a given time pattern
    
    :param time_pattern: input time pattern to add offset
    :type time_pattern: list 
    :param offset: offset to add
    :return: resulting pattern with offset
    """
    offset = offset % len(time_pattern)  # Handle offsets greater than the length
    return time_pattern[-offset:] + time_pattern[:-offset]


def add_instant_variation(time_pattern: list, noise_level: float = 0.1, seed=1):
    """
    Add instant variation to a given time pattern
     
    :param time_pattern: input time pattern to add variation on each instant
    :type time_pattern: list 
    :param noise_level: noise level/range. Default to 0.1
    :type noise_level: float
    :param seed: seed for noise
    :return: resulting pattern with variation
    """

    # Define the seed
    random.seed(seed)
    # Limit the value between 0 and 1
    return [min(max(x + random.uniform(-noise_level, noise_level), 0), 1) for x in time_pattern]


def combine_patterns(time_patterns: list[list], seed=1) -> list:
    """
    Combine a set of time_patterns randomly

    :param time_patterns: list of list
    :param seed: randomness seed
    :return: combined time pattern
    :rtype: list
    """
    # Get the length of the time_patterns
    length = len(time_patterns[0])
    # Define the seed
    random.seed(seed)

    # Check that all time_patterns are the same length
    if not all(len(pattern) == length for pattern in time_patterns):
        raise ValueError("All time_patterns must be the same length")

    # Create a new pattern with the same length
    new_pattern = []
    for i in range(length):
        # For each instant, choose a pattern randomly and take its value for that instant
        pattern = random.choice(time_patterns)
        new_pattern.append(pattern[i])

    return new_pattern


def smooth_edgy_values(time_pattern: list, threshold=0.20, smoothing_factor=0.20):
    """
    Smooth edgy values on an input time pattern
    :param time_pattern: input time pattern to add variation on each instant
    :type time_pattern: list
    :param threshold: threshold to consider a value edgy compared to previous value. Default to 0.20
    :param smoothing_factor: smoothing factor. Default to 0.20
    :return: smoothed input time pattern
    """
    # Start with the first value
    smoothed = [time_pattern[0]]
    # Iter over the length of the pattern from 1
    for i in range(1, len(time_pattern)):
        # Calculate difference between current and previous values
        diff = time_pattern[i] - time_pattern[i - 1]
        if abs(diff) > threshold:
            # If the difference is too large, smooth the value
            smoothed_value = smoothed[-1] + diff * smoothing_factor
            smoothed.append(smoothed_value)
        else:
            # If the difference is not too large, keep the original value
            smoothed.append(time_pattern[i])
    return smoothed
