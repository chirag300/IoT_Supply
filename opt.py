import argparse
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from data.gen_map import get_map

# Parameters and Indices
N = 11  # Number of Points (node 0 is warehouse, 10 stops)
K = 4  # Number of Products
np.random.seed(6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--map-data-path', default="data", help="Path to get map data from")
    parser.add_argument('-c', '--cluster', action="store_true", help="use clustered map instead of random map")
    parser.add_argument('-n', default=10, type=int, help="number of delivery points (make sure to generate this map first before trying to use)")
    args = parser.parse_args()

    # Base travel distances in km between specific stop pairs. Symmetric matrix
    base_travel_distance = get_map(args.map_data_path, args.cluster, args.n)

    # Calculate base travel times in hours
    base_travel_time = base_travel_distance / 30  # Base travel time, assuming average speed of 30 km/h
    base_delay_time = 0  # Base delay time at stops in seconds

    travel_times = base_travel_time * np.random.uniform(0.5, 1.5, size=base_travel_time.shape)  # +- 50% travel time
    delays = np.where(np.random.random(size=N) < 0.1, np.random.uniform(base_delay_time + 0.1, base_delay_time + 0.2, size=N), base_delay_time)  # 0.1 to 0.2 hours of delay in 10% of the cases

    print(travel_times)
    print(delays)

    base_temperature = {'Apples': 5, 'Bananas': 13, 'Tomatoes': 10, 'xyz': 8}
    required_temperature = [5, 13, 10, 8]
    temperature_profile = [base_temperature['Apples'] * (1 + np.random.uniform(-0.05, 0.05)),
                        base_temperature['Bananas'] * (1 + np.random.uniform(-0.05, 0.05)),
                        base_temperature['Tomatoes'] * (1 + np.random.uniform(-0.05, 0.05)),
                        base_temperature['xyz'] * (1 + np.random.uniform(-0.05, 0.05))]

    # Q = np.random.uniform(0.5, 1, K)
    # P = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    # alpha = 0.5

    model = gp.Model('Perishable Produce Transportation')
    x = model.addVars(N, N, vtype=GRB.BINARY, name="x")
    u = model.addVars(N, lb=0, ub=N-1, vtype=GRB.CONTINUOUS)

    model.addConstr(sum(x[0, j] for j in range(1, N)) == 1)  # This says we still have to return to node 0 (warehouse) at the end
    model.addConstr(sum(x[i, 0] for i in range(1, N)) == 1)  # Aren't these the same as the rest of the 1 incomring + 1 outgoing edge constraints?
    model.addConstrs((sum(x[i, j] for j in range(N) if j != i) == 1 for i in range(1, N)))
    model.addConstrs((sum(x[j, i] for j in range(N) if j != i) == 1 for i in range(1, N)))
    model.addConstrs((x[i, i] == 0 for i in range(N)))
    model.addConstrs((u[i] - u[j] + N * x[i, j] <= N - 1 for i in range(1, N) for j in range(1, N) if i != j))  # No subcycles constraint

    cost_term = sum((travel_times[i][j] + delays[i]) * x[i, j] for i in range(N) for j in range(N))
    temperature_penalty = sum(
        x[i, j] * sum(abs(temperature_profile[k] - required_temperature[k]) for k in range(K))
        for i in range(N) for j in range(N)
    )

    model.setObjective(cost_term + 0.05 * temperature_penalty, GRB.MINIMIZE)  # Temperature is only being used as a penalty and not a constraint, which is different from the paper. This is a Lagrangian relaxation of the original problem

    model.optimize()

    # Save results
    optimized_route = [0]
    if model.status == GRB.OPTIMAL:
        print("\nOptimal objective value (Total travel time in hours):", model.objVal)
        next = -1
        for i in range(N):
            if x[0, i].X == 1:
                next = i
                break
        optimized_route.append(next)
        while(next != 0):
            for i in range(N):
                if x[next, i].X == 1:
                    next = i
                    break
            optimized_route.append(next)
        
        print("Route is: " + " -> ".join(map(str, optimized_route)))
    else:
        print("No optimal solution found.")
        optimized_route = []
