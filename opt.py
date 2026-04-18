import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Parameters and Indices
n = 11  # Number of Points (node 0 is warehouse, 10 stops)
K = 4  # Number of Products
np.random.seed(6)

TRAIN_DATASET_LEN = 2000  # Number of dataset entries
# K = 3  # Number of products
RANDOM_LOCATIONS = "data/random_locations.npz"
CLUSTERED_LOCATIONS = "data/clustered_locations.npz"

# Base travel distances in km between specific stop pairs. Symmetric matrix
base_travel_distance = np.load(RANDOM_LOCATIONS)['n10']

# Calculate base travel times in hours
base_travel_time = base_travel_distance / 30  # Base travel time, assuming average speed of 30 km/h
base_delay_time = 0  # Base delay time at stops in seconds

travel_times = base_travel_time * np.random.uniform(0.5, 1.5, size=base_travel_time.shape)  # +- 50% travel time
delays = np.where(np.random.random(size=n) < 0.1, np.random.uniform(base_delay_time + 0.1, base_delay_time + 0.2, size=n), base_delay_time)  # 0.1 to 0.2 hours of delay in 10% of the cases

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
x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
u = model.addVars(n, lb=0, ub=n-1, vtype=GRB.CONTINUOUS)

model.addConstr(sum(x[0, j] for j in range(1, n)) == 1)  # This says we still have to return to node 0 (warehouse) at the end
model.addConstr(sum(x[i, 0] for i in range(1, n)) == 1)  # Aren't these the same as the rest of the 1 incomring + 1 outgoing edge constraints?
model.addConstrs((sum(x[i, j] for j in range(n) if j != i) == 1 for i in range(1, n)))
model.addConstrs((sum(x[j, i] for j in range(n) if j != i) == 1 for i in range(1, n)))
model.addConstrs((x[i, i] == 0 for i in range(n)))
model.addConstrs((u[i] - u[j] + n * x[i, j] <= n - 1 for i in range(1, n) for j in range(1, n) if i != j))  # No subcycles constraint

cost_term = sum((travel_times[i][j] + delays[i]) * x[i, j] for i in range(n) for j in range(n))
temperature_penalty = sum(
    x[i, j] * sum(abs(temperature_profile[k] - required_temperature[k]) for k in range(K))
    for i in range(n) for j in range(n)
)

model.setObjective(cost_term + 0.05 * temperature_penalty, GRB.MINIMIZE)  # Temperature is only being used as a penalty and not a constraint, which is different from the paper. This is a Lagrangian relaxation of the original problem

model.optimize()

# Save results
optimized_route = [0]
if model.status == GRB.OPTIMAL:
    print("\nOptimal objective value (Total travel time in hours):", model.objVal)
    next = -1
    for i in range(n):
        if x[0, i].X == 1:
            next = i
            break
    optimized_route.append(next)
    while(next != 0):
        for i in range(n):
            if x[next, i].X == 1:
                next = i
                break
        optimized_route.append(next)
    
    print("Route is: " + " -> ".join(map(str, optimized_route)))
else:
    print("No optimal solution found.")
    optimized_route = []
