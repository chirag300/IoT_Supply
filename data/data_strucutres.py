import math

import numpy as np
import torch


INIT_TRUCK_TEMP = 20
class TruckState:
    """
    State of the truck, and other statistics.
    """
    def __init__(self, location: int, temp: float = INIT_TRUCK_TEMP):
        # State
        self.temp = temp            # Temperature of the truck
        self.location = location    # Location of the truck (which vertex)

        # TODO: Other statistics if needed


class Produce:
    """
    Data structure representing one produce/crate type.
    """
    def __init__(
            self,
            name: str,
            quantity: float,
            initial_shelf_life: float,
            shelf_life_requirement: float,
            destination: int,
            activation_energy: float | None = None,
            pre_exponential_constant: float | None = None,
            initial_q10_rate: float | None = None,             # Rate of decay at 0C
        ):
        self.name = name
        self.quantity = quantity
        self.initial_shelf_life = initial_shelf_life
        self.shelf_life_requirement = shelf_life_requirement
        self.destination = destination
        self.activation_energy = activation_energy
        self.pre_exponential_constant = pre_exponential_constant
        self.initial_q10_rate = initial_q10_rate
        self.reset()

    def reset(self) -> None:
        self.shelf_life = self.initial_shelf_life
        self.delivered = False

    def shelf_life_update(self, temp_C: float, elapsed_time: float) -> None:
        # Arrhenius equation
        if self.activation_energy is not None and self.pre_exponential_constant is not None:
            decay_rate = self.pre_exponential_constant * math.exp(
                -self.activation_energy / (8.314462618 * (temp_C + 273.15))
            )
            self.shelf_life -= decay_rate * elapsed_time
        # Q10 model
        elif self.initial_q10_rate is not None:
            decay_rate = self.initial_q10_rate * (2.0 ** (temp_C / 10.0))
            self.shelf_life -= decay_rate * elapsed_time
        # Default linear model
        else:
            self.shelf_life -= elapsed_time


class DeliveryState:
    """
    State used in DQN model.
    """
    def __init__(
            self,
            dist_matrix: np.ndarray,    # n x n symmetric matrix. Map of destinations.
            produce: list[Produce],     # Produce to deliver.
        ):
        self.dist_matrix = dist_matrix
        self.produce = produce
        self.num_stops: int = self.dist_matrix.shape[0]  # NOTE: Including warehouse
        self.state_space = (self.num_stops ** 2) + self.num_stops + 2 + (7 * len(self.produce))
        self.reset()

    def reset(self) -> None:
        for p in self.produce:
            p.reset()
        self.visited = [0 for _ in range(self.num_stops)]
        self.visited[0] = 1
        self.truck_state = TruckState(0, INIT_TRUCK_TEMP)

    def get_state(self) -> torch.Tensor:
        out = torch.empty(self.state_space, dtype=torch.float32)
        idx = self.num_stops ** 2
        out[: idx] = torch.from_numpy(self.dist_matrix).flatten()
        out[idx : idx + self.num_stops] = torch.tensor(self.visited, dtype=torch.float32)
        idx += self.num_stops
        out[idx] = float(self.truck_state.location)
        idx += 1
        out[idx] = float(self.truck_state.temp)

        idx += 1
        for p in self.produce:
            out[idx] = float(p.shelf_life - p.shelf_life_requirement)  # Shelf life remaining
            out[idx + 1] = float(p.destination)
            out[idx + 2] = float(p.delivered)
            out[idx + 3] = p.quantity
            out[idx + 4] = p.activation_energy if p.activation_energy else 0.0
            out[idx + 5] = p.pre_exponential_constant if p.pre_exponential_constant else 0.0
            out[idx + 6] = p.initial_q10_rate if p.initial_q10_rate else 0.0
            idx += 7
        return out
