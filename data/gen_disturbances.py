import copy

import numpy as np

from .data_strucutres import Produce


def gen_dist_disturbances(dist_matrix: np.ndarray) -> np.ndarray:
    """Create a new copy of the symmetric distance matrix after applying uncertainties."""
    # TODO: Use probability distributions and/or peak/low time to simulate traffic. Or adjust this in real time somehow.
    traffic_adjusted_distances = dist_matrix.copy()

    return traffic_adjusted_distances

def gen_produce_disturbances(produce: list[Produce]) -> list[Produce]:
    """Create a new copy of produce list after applying uncertainties in the shelf life decay rates."""
    # TODO: Use probability distributions to apply uncertainties in activation energy and pre-exponential constant.
    rate_adjusted_produce = copy.deepcopy(produce)

    return rate_adjusted_produce
