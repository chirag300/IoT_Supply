import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from data.gen_map import get_map
from data.gen_training_data import get_delivery_plan, get_inventory, get_map, get_produce, get_shelf_life, get_stop_pairs

# =========================== ENVIRONMENT CLASS =========================== #
class DeliveryEnv:
    def __init__(
            self,
            stops_data: pd.DataFrame,
            produce_data: pd.DataFrame,
            dist_matrix,
            delivery_plan,
            initial_inventory,
            initial_shelf_life,
            max_steps=50
        ):
        self.stops_data = stops_data  # TODO: This is a static list that does not get updated, so we are not training anything useful. It is like giving the truck 200 different ways to go from A to B, of course it always just picks the shortest one.
        self.produce_data = produce_data  # TODO: This is not used anywhere.
        self.dist_matrix = dist_matrix  # TODO: Can use this to dynamically generate the disturbances. Currently not used anywhere
        self.max_steps = max_steps
        self.all_stops = sorted(set(stops_data['stop_i'].unique()) | set(stops_data['stop_j'].unique()))
        self.n_stops = len(self.all_stops)
        # self.n_stops = self.map.shape[0]
        # self.all_stops = list(range(self.n_stops))
        self.action_space_size = self.n_stops
        self.delivery_plan = delivery_plan
        self.initial_inventory = initial_inventory
        self.initial_shelf_life = initial_shelf_life
        self.reset()

    def reset(self):
        self.warehouse_location = 0
        self.current_location = self.warehouse_location
        self.current_inventory = self.initial_inventory
        self.elapsed_time = 0
        self.shelf_life_remaining = self.initial_shelf_life
        self.steps = 0
        self.visited_stops = {self.warehouse_location}
        self.route_log = []
        self.log_initial_stop()
        return self.get_state()

    def get_state(self):
        return [
            self.current_location,
            self.current_inventory['Apples'],
            self.current_inventory['Bananas'],
            self.current_inventory['Tomatoes'],
            self.current_inventory['xyz'],
            self.shelf_life_remaining['Apples'],
            self.shelf_life_remaining['Bananas'],
            self.shelf_life_remaining['Tomatoes'],
            self.shelf_life_remaining['xyz'],
            self.elapsed_time,
        ]

    def get_nearest_stops(self):
        """Fetch all unvisited neighbouring stops."""
        unvisited_routes = self.stops_data[
            ((self.stops_data['stop_i'] == self.current_location) & (~self.stops_data['stop_j'].isin(self.visited_stops))) |
            ((self.stops_data['stop_j'] == self.current_location) & (~self.stops_data['stop_i'].isin(self.visited_stops)))
        ]
        if unvisited_routes.empty:
            return []

        unvisited_routes = unvisited_routes.copy()
        unvisited_routes['next_stop'] = np.where(
            unvisited_routes['stop_i'] == self.current_location,
            unvisited_routes['stop_j'],
            unvisited_routes['stop_i']
        )
        return unvisited_routes[['next_stop', 'travel_distance_km', 'travel_time_hours', 'delay_time_hours']].values.tolist()

    def get_temperature_adjustment(self):
        """
        Compute the necessary temperature adjustment to maintain ideal storage conditions.
        """

        # Define ideal temperatures
        ideal_temperatures = {'Apples': 2, 'Bananas': 5, 'Tomatoes': 8, 'xyz': 7}

        # Assume the truck currently has a storage temperature (simulated sensor data)
        # TODO: This should be part of the environment's state, 
        current_temperatures = {
            'Apples': np.random.uniform(0, 4),  # Simulated sensor reading
            'Bananas': np.random.uniform(12, 14),
            'Tomatoes': np.random.uniform(7, 10),
            'xyz': np.random.uniform(0, 10)
        }

        # Compute temperature deviation for each produce type
        temp_deviation = {p : ideal_temperatures[p] - current_temperatures[p] for p in ideal_temperatures}

        # Compute the average temperature adjustment needed
        avg_temp_adjustment = sum(temp_deviation.values()) / len(temp_deviation)

        return avg_temp_adjustment  # Negative means decrease temp, positive means increase temp


    def select_stop(self, nearest_stops):
        """Select the next stop based on weighted scoring with priority to distance and delay."""
        best_stop = None
        best_score = float('inf')  # Smaller scores are better for prioritizing distance and delay

        for stop in nearest_stops:
            next_stop, travel_distance_km, travel_time, delay_time = stop
            delivery = self.delivery_plan.get(next_stop, {})
            delivery_score = sum(min(self.current_inventory[p], v) for p, v in delivery.items())
            avg_shelf_life = np.mean([self.shelf_life_remaining[p] for p in delivery if delivery[p] > 0])

            # Weighted scoring: prioritize distance, delay, then delivery quantity, then shelf life
            delay_penalty = 50 * (delay_time > 5) + 15 * delay_time  # Large penalty for delays > 5 hours
            score = (10 * travel_distance_km) + delay_penalty - (2 * delivery_score) - avg_shelf_life

            # print(f"Evaluating stop {next_stop}: Distance = {travel_distance_km}, Delay = {delay_time}, "
            #     f"Delivery Score = {delivery_score}, Shelf Life = {avg_shelf_life}, Score = {score}")

            if score < best_score:  # Lower score is better
                best_score = score
                best_stop = stop

        return best_stop

    # TODO: Fix this, we are not using the agent's decision `action`. We should be going to whichever destination the agent chose to go.
    # We should not be greedily picking the nearest stop.
    def step(self, action):
        nearest_stops = self.get_nearest_stops()
        if not nearest_stops:
            if self.current_location != self.warehouse_location:
                self.return_to_warehouse()
            return self.get_state(), 0, True

        # Select best stop based only on shelf life (unchanged)
        best_stop = self.select_stop(nearest_stops)
        if not best_stop:
            if self.current_location != self.warehouse_location:
                self.return_to_warehouse()
            return self.get_state(), 0, True

        next_stop, _, travel_time, delay_time = best_stop

        # Update environment state
        total_time = travel_time + delay_time
        self.elapsed_time += total_time
        self.steps += 1
        self.update_shelf_life()

        # Perform deliveries
        delivery = self.delivery_plan.get(next_stop, {})
        delivered_count = sum(self.update_inventory(p, amt) for p, amt in delivery.items())

        # Compute the temperature adjustment required
        temp_adjustment = self.get_temperature_adjustment()

        # Print temperature adjustment information
        if temp_adjustment < 0:
            print(f"At Stop {next_stop}: Truck should DECREASE temperature by {abs(temp_adjustment):.2f}°C")
        elif temp_adjustment > 0:
            print(f"At Stop {next_stop}: Truck should INCREASE temperature by {temp_adjustment:.2f}°C")
        else:
            print(f"At Stop {next_stop}: No temperature adjustment needed.")

        # Reward calculation (unchanged)
        reward = self.calculate_reward(total_time, delivered_count)

        # Move to next stop
        self.current_location = next_stop
        self.visited_stops.add(next_stop)
        self.log_route(next_stop, delivery, reward)

        # Check if all stops are visited
        if len(self.visited_stops) == self.n_stops:
            self.return_to_warehouse()

        done = self.check_done()
        return self.get_state(), reward, done


    def return_to_warehouse(self):
        """Move back to the warehouse and add travel time."""
        if self.current_location != self.warehouse_location:
            travel_info = self.stops_data[
                (self.stops_data['stop_i'] == self.current_location) & (self.stops_data['stop_j'] == self.warehouse_location)
                | (self.stops_data['stop_j'] == self.current_location) & (self.stops_data['stop_i'] == self.warehouse_location)
            ]
            if not travel_info.empty:
                travel_time = travel_info.iloc[0]['travel_time_hours']
                self.elapsed_time += travel_time
                self.log_route(self.warehouse_location, {}, 0)  # No penalty for returning to warehouse
            self.current_location = self.warehouse_location

    def update_inventory(self, produce, amount):
        """Update inventory after delivery."""
        delivered = min(self.current_inventory.get(produce, 0), amount)
        self.current_inventory[produce] -= delivered
        return delivered

    def update_shelf_life(self):
        """Update shelf life based on time elapsed and decay rate."""
        decay_rate = 0.05  # Base decay factor
        for produce in self.shelf_life_remaining:
            self.shelf_life_remaining[produce] -= self.elapsed_time * decay_rate
            self.shelf_life_remaining[produce] = max(0, self.shelf_life_remaining[produce])

    def calculate_reward(self, total_time, delivered_produce_count):
        """Calculate reward with separate penalties for travel and delivery rewards."""
        # TODO: There is a possibility that the agent used random action, and goes to an illegal stop. That should be punished with -infinity and instant termination
        delivery_reward = delivered_produce_count * 10  # Reward for each item delivered
        travel_penalty = total_time * 0.1  # Penalty proportional to travel and delay time
        return delivery_reward - travel_penalty

    def check_done(self):
        """Determine if the episode should terminate."""
        return (
            self.steps >= self.max_steps
            or self.current_location == self.warehouse_location
            and len(self.visited_stops) == self.n_stops
            or all(v == 0 for v in self.current_inventory.values())
            or all(v <= 0 for v in self.shelf_life_remaining.values())
        )

    # TODO: This is not used anywhere
    def handle_no_stops(self):
        if self.current_location != self.warehouse_location:
            self.current_location = self.warehouse_location
        return self.get_state(), -10, True  # Penalty for no valid stops

    def log_route(self, stop, delivery, reward):
        self.route_log.append({'Stop': stop, 'Delivered': delivery, 'Reward': reward})

    def log_initial_stop(self):
        self.route_log.append({'Stop': self.warehouse_location, 'Reward': 0})


# =========================== DQN AGENT =========================== #
class DQNAgent:
    def __init__(self, state_size, action_size, model: nn.Module = None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.0005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.build_model().to(self.device)
        self.target_model = self.build_model().to(self.device)
        if model:
            self.model.load_state_dict(model.state_dict())
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # Multi-layer perceptron
    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return torch.argmax(self.model(state)).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        max_next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# =========================== TRAINING LOOP =========================== #
def train_dqn(train_env: DeliveryEnv, initial_model: nn.Module = None, episodes=500, target_update=10, batch_size=64):
    agent = DQNAgent(len(train_env.get_state()), train_env.action_space_size, initial_model)
    rewards = []  # Track rewards for visualization
    elapsed_times = []  # Track elapsed times

    for episode in range(episodes):
        # Update the target model every few episodes
        if episode % target_update == 0:
            agent.update_target_model()

        state = train_env.reset()
        total_reward, done = 0, False
        elapsed_time = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = train_env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            elapsed_time = train_env.elapsed_time  # Track final elapsed time

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        rewards.append(total_reward)
        elapsed_times.append(elapsed_time)

    return agent, rewards, elapsed_times


def save_model(agent: DQNAgent, file_path: str = "dqn_agent.pth"):
    """Save the trained model to a file."""
    torch.save(agent.model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(agent: DQNAgent, file_path: str = "dqn_agent.pth"):
    """Load a trained model from a file."""
    if os.path.exists(file_path):
        agent.model.load_state_dict(torch.load(file_path))
        agent.update_target_model()
        print(f"Model loaded from {file_path}")
    else:
        print(f"Model file {file_path} does not exist.")

def train_dqn_agent(train_env: DeliveryEnv, episodes=500, target_update=10, batch_size=64, save_path="dqn_agent.pth"):
    # TODO: Should be trained on the same map, but also with different delays/uncertainties
    # Do this by placing the below in a loop, while keeping the model. So, agent can learn how to navigate many different possible delays

    # Initialize environment
    train_env.reset()

    # Train DQN agent
    print("Starting DQN training...\n")
    agent, rewards, elapsed_times = train_dqn(train_env, None, episodes, target_update, batch_size)

    # Update target model after training
    agent.update_target_model()

    # Save the trained model
    save_model(agent, save_path)

    # ================= PRINT BEST ROUTE OF TRAINING DATASET ================= #
    print("\n--- Best Route Found In Training Dataset ---")
    train_env.reset()
    done = False
    while not done:
        action = agent.act(train_env.get_state())
        _, _, done = train_env.step(action)

    # Display route log
    for log in train_env.route_log:
        print(log)
    print(f"Total time of route: {train_env.elapsed_time} h")

def run_model(test_env: DeliveryEnv, save_path: str = "dqn_agent.pth"):
    # Load previously trained RL agent and run it to see what path it would choose
    agent = DQNAgent(len(test_env.get_state()), test_env.action_space_size)
    load_model(agent, save_path)

    print("\n--- Best Route Found ---")
    test_env.reset()
    done = False
    while not done:
        action = agent.act(test_env.get_state())
        _, _, done = test_env.step(action)

    # Display route log
    for log in test_env.route_log:
        print(log)
    print(f"Total time of route: {test_env.elapsed_time} h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save-path', default="dqn_agent.pth", help="Path to save model to")
    parser.add_argument('-t', '--train', action='store_true')
    args = parser.parse_args()

    # Train the agent
    train_env = DeliveryEnv(get_stop_pairs("data"), get_produce(), get_map(), get_delivery_plan(), get_inventory(), get_shelf_life())
    if args.train:
        train_dqn_agent(train_env, episodes=100)

    # Use a real test dataset that's different than the one used in training (generates new variations from same base map)
    test_env = DeliveryEnv(get_stop_pairs("data"), get_produce(), get_map(), get_delivery_plan(), get_inventory(), get_shelf_life())
    run_model(test_env)
