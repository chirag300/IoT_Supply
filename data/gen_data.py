import argparse
import pickle

import numpy as np

from .data_strucutres import Produce

# ==================================== GENERATE MAP ====================================
# Generate a map of locations in 2D Euclidean plane to represent stores/customers (distances in km)
# (0,0) is the warehouse
# To mimic that the warehouse is "outside the city", only generate points 3<x<30, 3<y<30
MIN_DIST = 3
MAX_DIST = 30
# For clustered locations, a cluster is a 3x3 box. Min distance between cluster borders is 3.
CLUSTER_SIZE = 3
CLUSTER_DIST = 3

RANDOM_LOCATIONS_FILENAME = "random_locations.npz"
CLUSTERED_LOCATIONS_FILENAME = "clustered_locations.npz"

def gen_dist_matrix(points: np.ndarray):
    """Generate a symmetrical (euclidean) distance matrix from a list of points."""
    # Using broadcasting: (N, 1, 2) - (1, N, 2) results in pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # Calculate Euclidean distance: sqrt(dx^2 + dy^2)
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    return dist_matrix

def gen_random_locations(num_locations: int):
    """Generate random points."""
    print("Generating random locations")
    points = np.array([[0, 0]])
    points = np.append(points, np.random.uniform(MIN_DIST, MAX_DIST, size=(num_locations, 2)), axis=0)

    return points

def gen_clustered_locations(locations_per_cluster: list[int]):
    """Generate a map with locations_per_cluster[i] number of points at cluster i."""
    num_clusters = len(locations_per_cluster)
    centers = []
    attempts = 0
    retries = 0
    print("Generating clusters", end="", flush=True)
    while retries < 100:
        while len(centers) < num_clusters and attempts < 1000 + num_clusters:
            # Generate cluster center so that the whole cluster is within the bounds
            new_center = np.random.uniform(MIN_DIST + CLUSTER_SIZE/2, MAX_DIST - CLUSTER_SIZE/2, size=2)
            # Make sure it's far enough from all other clusters
            if all(np.min(np.abs(new_center - c)) >= CLUSTER_SIZE+CLUSTER_DIST for c in centers):
                centers.append(new_center)
            attempts += 1

        if(len(centers) != num_clusters):
            retries += 1
            if(retries % 5 == 0):
                print(".", end="", flush=True)
            centers = []
            attempts = 0
        else:
            break
    print()

    if(len(centers) != num_clusters):
        raise ValueError("Possibly too many clusters and cannot fit in defined region.")

    points = [np.array([0, 0])]  # Start with the origin
    for i in range(num_clusters):
        # Generate points within the cluster
        cluster_points = np.random.uniform(centers[i] - CLUSTER_SIZE/2, centers[i] + CLUSTER_SIZE/2, size=(locations_per_cluster[i], 2))
        points.append(cluster_points)

    all_points = np.vstack(points)

    return all_points

def get_map(save_path: str = "data", clustered: bool = False, n: int = 10) -> np.ndarray:
    """Load a pre-generated map with n stops (not including warehouse)."""
    return np.load(save_path + "/" + (CLUSTERED_LOCATIONS_FILENAME if clustered else RANDOM_LOCATIONS_FILENAME))[f"n{n}"]


# ==================================== GENERATE PRODUCE ====================================
PRODUCE_NAMES = ["apple", "banana", "carrot", "dragon fruit", "egg", "fig", "grape", "ham", "incaberry", "jujube", "kale", "lettuce"]

PRODUCE_FILENAME = "produce.pkl"

def gen_produce(num_produce_per_stop: list[int]) -> list[Produce]:
    """Generate num_produce_per_stop[i] types of produce to deliver to stop i."""
    print("Generating produce sets")
    produce = []
    for stop in range(len(num_produce_per_stop)):
        for i in range(num_produce_per_stop[stop]):
            shelf_life_hours = np.random.uniform(24, 336)    # 1 day to 2 weeks
            activation_energy = np.random.uniform(4e4, 7e4)
            produce.append(Produce(
                PRODUCE_NAMES[i % len(PRODUCE_NAMES)] + " " + str(i),  # Name of produce
                np.random.uniform(1, 10),                              # Quantity (kg, or $USD)
                shelf_life_hours,                                      # Initial shelf life (shelf_life_hours)
                max(0, shelf_life_hours - np.random.uniform(6, 120)),  # Shelf life requirement (shelf_life_hours)
                stop + 1,                                              # Destination
                activation_energy,                                     # Activation energy (J/mol)
                10 ** ((activation_energy/1e4 - 4)/3*4 + 7),           # Pre-exponential constant (-shelf_life_hours/h), between 1e7 and 1e11
            ))
    return produce

def get_produce(save_path: str = "data", n: int = 10) -> list[Produce]:
    """Load pre-generated set of produce for a map with n stops (not including warehouse)."""
    with open(save_path + "/" + PRODUCE_FILENAME, "rb") as file:
        produce_data = pickle.load(file)
    return produce_data[f"n{n}"]


# ==================================== GENERATE DATA MAIN SCRIPT ====================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save-path', default="data", help="Path to save map data to")
    args = parser.parse_args()

    # Currently just generates 1 map per case
    np.savez(args.save_path + "/" + RANDOM_LOCATIONS_FILENAME,
        n10 = gen_dist_matrix(gen_random_locations(10)),
        n20 = gen_dist_matrix(gen_random_locations(20)),
        n30 = gen_dist_matrix(gen_random_locations(30)),
    )

    # Currently just generates 1 map per case
    np.savez(args.save_path + "/" + CLUSTERED_LOCATIONS_FILENAME,
        n10 = gen_dist_matrix(gen_clustered_locations([2,3,5])),
        n20 = gen_dist_matrix(gen_clustered_locations([5,7,8])),
        n30 = gen_dist_matrix(gen_clustered_locations([3,6,8,13])),
    )

    # Currently just generates 1 produce set per case
    with open(args.save_path + "/" + PRODUCE_FILENAME, "wb") as file:
        produce_data = {
            "n10": gen_produce([np.random.randint(1,5) for _ in range(10)]),
            "n20": gen_produce([np.random.randint(1,5) for _ in range(20)]),
            "n30": gen_produce([np.random.randint(1,5) for _ in range(30)]),
        }
        pickle.dump(produce_data, file)
