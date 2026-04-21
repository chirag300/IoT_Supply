import numpy as np
import pandas as pd

from .gen_map import get_map

TRAIN_DATASET_LEN = 2000  # Number of dataset entries

RANDOM_LOCATIONS = "random_locations.npz"
CLUSTERED_LOCATIONS = "clustered_locations.npz"


def get_stop_pairs(save_path: str = "data", clustered: bool = False, n: int = 10):
    # Base travel distances between specific stop pairs. Symmetric matrix
    base_travel_distance = get_map(save_path, clustered, n)

    # Calculate base travel times in hours
    base_travel_time = base_travel_distance / 30  # Base travel time, assuming average speed of 30 km/h
    base_delay_time = 0  # Base delay time at stops in hours

    # Delay reasons list
    delay_reasons = ["Road Closed", "Traffic", "Accident", "Weather", "No Delay"]

    # Lists to store generated data
    travel_distance_pattern = []
    travel_time_pattern = []
    delay_time_pattern = []
    stops_i = []
    stops_j = []
    delay_events = []

    # Generate shelf life and quality factors
    # SL_0 = 700 * np.random.rand(K)  # Initial shelf life for each product
    # SL_r = 100 * np.random.rand(K)  # Required shelf life at destination for each product
    # Q = np.random.uniform(0.5, 1, K)  # Quality reduction factor for each product

    # NOTE: This only generates the dynamic variances upon 1 given map. This trained agent will not generalize well to using a new map/route.
    # To generalize to any route, the data given to the agent should not be just a stop pair, it should be the whole distance matrix
    for _ in range(TRAIN_DATASET_LEN):
        stop_i = np.random.randint(0, 11)  # Random stop
        stop_j = np.random.randint(0, 11)

        # Ensure stop_i != stop_j
        while stop_i == stop_j:
            stop_j = np.random.randint(0, 11)

        # Add some variation to travel times
        # TODO: Why do we keep track of distance? It has no effect on produce shelf life, maybe only fuel consumption
        distance = base_travel_distance[stop_i][stop_j] * np.random.uniform(0.5, 1.5)  # +- 50%
        time = base_travel_time[stop_i][stop_j] * np.random.uniform(0.5, 1.5)      # +- 50%

        # Stop delay
        if np.random.rand() < 0.1:  # Add random variation in 10% of cases
            delay = base_delay_time + np.random.uniform(0.1, 0.2)  # 0.1 to 0.2 hours of delay
            event = np.random.choice(delay_reasons[1:-1])  # Select a reason excluding "No Delay" and "Road Closed"
        else:
            delay = base_delay_time
            event = "No Delay"  # No significant delay

        # Append to lists
        stops_i.append(stop_i)
        stops_j.append(stop_j)
        travel_distance_pattern.append(distance)
        travel_time_pattern.append(time)
        delay_time_pattern.append(delay)
        delay_events.append(event)

    # Create a DataFrame with stop pairs and generated values
    stop_pairs_pattern = pd.DataFrame({
        'stop_i': stops_i,
        'stop_j': stops_j,
        'travel_distance_km': travel_distance_pattern,
        'travel_time_hours': travel_time_pattern,
        'delay_time_hours': delay_time_pattern,
        'delay_event': delay_events
    })

    # Modify the stop_pairs_pattern DataFrame to include a large delay for route 5 -> 4
    stop_pairs_pattern.loc[(stop_pairs_pattern['stop_i'] == 5) & (stop_pairs_pattern['stop_j'] == 4), 'delay_time_hours'] = 1  # 1 hour delay
    stop_pairs_pattern.loc[(stop_pairs_pattern['stop_i'] == 5) & (stop_pairs_pattern['stop_j'] == 4), 'delay_event'] = 'Road Block'
    # Modify the symmetrical route 4 -> 5 as well
    stop_pairs_pattern.loc[(stop_pairs_pattern['stop_i'] == 4) & (stop_pairs_pattern['stop_j'] == 5), 'delay_time_hours'] = 1  # 1 hour delay
    stop_pairs_pattern.loc[(stop_pairs_pattern['stop_i'] == 4) & (stop_pairs_pattern['stop_j'] == 5), 'delay_event'] = 'Road Block'

    return stop_pairs_pattern

def get_produce():
    # Define the types of produce and their typical characteristics
    produce_types = ['Apples', 'Bananas', 'Tomatoes', 'xyz']

    # Define base shelf life (in hours) for each produce type
    base_shelf_life = {
        'Apples': 720,   # 30 days
        'Bananas': 168,  # 7 days
        'Tomatoes': 336, # 14 days
        'xyz': 200
    }

    temperature_range = {
        'Apples': (0, 4),
        'Bananas': (12, 14),
        'Tomatoes': (7, 10),
        'xyz': (0, 10)
    }

    # Lists to store generated data
    produce_type_list = []
    weight_list = []
    shelf_life_list = []
    transport_temperature_list = []

    for _ in range(TRAIN_DATASET_LEN):
        produce = np.random.choice(produce_types)

        weight = np.random.uniform(0.5, 20)

        # Shelf life with some variation (up to +-10%)
        shelf_life = base_shelf_life[produce] * np.random.uniform(0.9, 1.1)

        # Transportation temperature within defined range for the selected produce
        transport_temperature = np.random.uniform(*temperature_range[produce])

        # Append to lists
        produce_type_list.append(produce)
        weight_list.append(weight)
        shelf_life_list.append(shelf_life)
        transport_temperature_list.append(transport_temperature)

    # Create a DataFrame with the generated data
    produce_data = pd.DataFrame({
        'produce_type': produce_type_list,
        'weight_kg': weight_list,
        'shelf_life_hours': shelf_life_list,
        'transport_temperature_C': transport_temperature_list
    })

    return produce_data

def get_delivery_plan():
    return {
        1: {'Apples': 10, 'Bananas': 5, 'Tomatoes': 5, 'xyz': 0},
        2: {'Apples': 5, 'Bananas': 10, 'Tomatoes': 5, 'xyz': 5},
        3: {'Apples': 5, 'Bananas': 5, 'Tomatoes': 10, 'xyz': 5},
        4: {'Apples': 10, 'Bananas': 5, 'Tomatoes': 0, 'xyz': 10},
        5: {'Apples': 7, 'Bananas': 7, 'Tomatoes': 6, 'xyz': 5},
        6: {'Apples': 8, 'Bananas': 5, 'Tomatoes': 7, 'xyz': 5},
        7: {'Apples': 6, 'Bananas': 8, 'Tomatoes': 6, 'xyz': 5},
        8: {'Apples': 5, 'Bananas': 5, 'Tomatoes': 8, 'xyz': 7},
        9: {'Apples': 10, 'Bananas': 5, 'Tomatoes': 5, 'xyz': 5},
        10: {'Apples': 10, 'Bananas': 6, 'Tomatoes': 6, 'xyz': 6},
    }

def get_inventory():
    return {'Apples': 76, 'Bananas': 61, 'Tomatoes': 58, 'xyz': 53}

def get_shelf_life():
    return {'Apples': 720, 'Bananas': 168, 'Tomatoes': 336, 'xyz': 200}
