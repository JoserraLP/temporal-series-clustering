import numpy as np

from temporal_series_clustering.patterns.generators import predictor_city, \
    create_simulation_specific_weekday, add_instant_variation, add_offset, smooth_edgy_values, combine_patterns


def generate_patterns(weekday: str, total_num=50):
    # Get the simulations of each predictor
    a_simulation = create_simulation_specific_weekday(predictor_city, place_id='a', weekday=weekday)
    b_simulation = create_simulation_specific_weekday(predictor_city, place_id='b', weekday=weekday)
    c_simulation = create_simulation_specific_weekday(predictor_city, place_id='c', weekday=weekday)
    d_simulation = create_simulation_specific_weekday(predictor_city, place_id='d', weekday=weekday)
    e_simulation = create_simulation_specific_weekday(predictor_city, place_id='e', weekday=weekday)
    f_simulation = create_simulation_specific_weekday(predictor_city, place_id='f', weekday=weekday)

    predictors_output = [a_simulation, b_simulation, c_simulation, d_simulation, e_simulation, f_simulation]

    # 6 patterns
    # Get patterns with noise
    instant_noise = 0.05
    a_noise = add_instant_variation(a_simulation, noise_level=instant_noise)
    b_noise = add_instant_variation(b_simulation, noise_level=instant_noise)
    c_noise = add_instant_variation(c_simulation, noise_level=instant_noise)
    d_noise = add_instant_variation(d_simulation, noise_level=instant_noise)
    e_noise = add_instant_variation(e_simulation, noise_level=instant_noise)
    f_noise = add_instant_variation(f_simulation, noise_level=instant_noise)

    predictors_output.extend([a_noise, b_noise, c_noise, d_noise, e_noise, f_noise])

    # 12 patterns
    # Less noise
    instant_noise = 0.01
    a_noise = add_instant_variation(a_simulation, noise_level=instant_noise)
    b_noise = add_instant_variation(b_simulation, noise_level=instant_noise)
    c_noise = add_instant_variation(c_simulation, noise_level=instant_noise)
    d_noise = add_instant_variation(d_simulation, noise_level=instant_noise)
    e_noise = add_instant_variation(e_simulation, noise_level=instant_noise)
    f_noise = add_instant_variation(f_simulation, noise_level=instant_noise)

    predictors_output.extend([a_noise, b_noise, c_noise, d_noise, e_noise, f_noise])

    # 18 patterns

    # Add offset for original and noise patterns

    np.random.seed(1)
    offset = np.random.randint(-2, 2)
    a_offset = add_offset(a_simulation, offset=offset)
    offset = np.random.randint(-2, 2)
    b_offset = add_offset(b_simulation, offset=offset)
    offset = np.random.randint(-2, 2)
    c_offset = add_offset(c_simulation, offset=offset)
    offset = np.random.randint(-2, 2)
    d_offset = add_offset(d_simulation, offset=offset)
    offset = np.random.randint(-2, 2)
    e_offset = add_offset(e_simulation, offset=offset)
    offset = np.random.randint(-2, 2)
    f_offset = add_offset(f_simulation, offset=offset)

    predictors_output.extend([a_offset, b_offset, c_offset, d_offset, e_offset, f_offset])

    # 24 patterns

    # Combine remaining patterns
    pattern_combinations = []
    for i in range(total_num - len(predictors_output)):
        # one more than the length as we want one more item due to range behavior
        pattern_combinations.append(smooth_edgy_values(combine_patterns(predictors_output, seed=i)))

    predictors_output.extend(pattern_combinations)

    return predictors_output[:total_num]


def generate_similar_patterns(weekday: str, total_num=21):
    # Get the simulations of each predictor
    a_simulation = create_simulation_specific_weekday(predictor_city, place_id='a', weekday=weekday)
    b_simulation = create_simulation_specific_weekday(predictor_city, place_id='b', weekday=weekday)
    c_simulation = create_simulation_specific_weekday(predictor_city, place_id='c', weekday=weekday)
    d_simulation = create_simulation_specific_weekday(predictor_city, place_id='d', weekday=weekday)
    e_simulation = create_simulation_specific_weekday(predictor_city, place_id='e', weekday=weekday)
    f_simulation = create_simulation_specific_weekday(predictor_city, place_id='f', weekday=weekday)

    predictors_output = [a_simulation, b_simulation, c_simulation, d_simulation, e_simulation, f_simulation,
                         a_simulation, b_simulation, c_simulation, d_simulation, e_simulation, f_simulation,
                         a_simulation, b_simulation, c_simulation, d_simulation, e_simulation, f_simulation]

    # Create a pattern that is half one and half other
    mid_point = len(a_simulation) // 2

    mid_point += 1 if len(a_simulation) % 2 else -1

    a_b_simulation = a_simulation[:mid_point] + b_simulation[mid_point:]
    c_b_simulation = c_simulation[:mid_point] + d_simulation[mid_point:]
    e_f_simulation = e_simulation[:mid_point] + f_simulation[mid_point:]

    predictors_output.extend([a_b_simulation, c_b_simulation, e_f_simulation])

    return predictors_output[:total_num]
