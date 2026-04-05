import numpy as np

# Generate a map of locations in 2D Euclidean plane to represent stores/customers (distances in km)
# (0,0) is the warehouse
# To mimic that stores/customers are more clustered together compared to the warehouse, only generate points 3<x<30, 3<y<30
MIN_DIST = 3
MAX_DIST = 30
# For clustered locations, a cluster is a 3x3 box. Min distance between cluster borders is 3.
CLUSTER_SIZE = 3
CLUSTER_DIST = 3

# Data set 1: Random
def gen_random_locations(num_locations: int):
    points = np.random.uniform(MIN_DIST, MAX_DIST, size=(num_locations, 2))
    points = np.append(points, [[0, 0]], axis=0)
    
    return points

def get_dist_matrix(points: np.ndarray):
    # Using broadcasting: (N, 1, 2) - (1, N, 2) results in pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # Calculate Euclidean distance: sqrt(dx^2 + dy^2)
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    return dist_matrix

# Currently just generates 1 scenario per case
np.savez("random_locations.npz",
    n10 = get_dist_matrix(gen_random_locations(10)),
    n20 = get_dist_matrix(gen_random_locations(20)),
    n30 = get_dist_matrix(gen_random_locations(30))
)

# Data set 2: local clusters
def gen_clustered_locations(locations_per_cluster: list[int]):
    num_clusters = len(locations_per_cluster)
    centers = []
    attempts = 0
    retries = 0
    print("Generating clusters", end="", flush=True)
    while retries < 100:
        while len(centers) < num_clusters and attempts < 1000+num_clusters:
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

# Currently just generates 1 scenario per case
np.savez("clustered_locations.npz",
    n10 = get_dist_matrix(gen_clustered_locations([2,3,5])),
    n20 = get_dist_matrix(gen_clustered_locations([5,7,8])),
    n30 = get_dist_matrix(gen_clustered_locations([3,6,8,13]))
)
