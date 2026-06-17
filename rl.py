import argparse
from collections import deque
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data.gen_data import get_map, get_produce
from data.gen_disturbances import gen_dist_disturbances, gen_produce_disturbances
from data.data_strucutres import DeliveryState, Produce, TruckState


# =========================== ENVIRONMENT CLASS =========================== #
GET_TRUCK_TEMP = 20
TRUCK_SPEED = 30 # Avg truck speed, km/h (not really important)
TRAVEL_TIME_PENALTY_WEIGHT = 0.1

class DeliveryEnv:
    """
    Environment of agent. Defines how the environment modifies state.
    """
    def __init__(
            self,
            dist_matrix: np.ndarray,            # TODO: Can use this to dynamically generate the disturbances. Currently not used anywhere
            produce: list[Produce],
            optimal_route: list[TruckState],    # Optimal route if no traffic/disturbances on the roads
        ):
        self.state = DeliveryState(dist_matrix, produce)
        self.optimal_route = optimal_route  # TODO: Currently unused. Could be used as a hint for the agent instead of doing random select, possibly speeds up training?
        self.reset()

    def reset(self) -> torch.Tensor:
        self.state.reset()
        self.route_log = [0]
        self.elapsed_time = 0.0
        self.stops_visited = 1
        return self.state.get_state()

    def step(self, next_stop: int) -> tuple[torch.Tensor, float, bool]:
        """
        Update the environment to simulate going to the next stop.
        Return: New state, reward, done
        """
        # Check if legal. If not legal, terminate with garbage next state.
        if not self.is_legal(next_stop):
            raise ValueError("This should never happen with action masking...")
            self.route_log.append(next_stop)  # For logging purposes
            return torch.zeros(self.state.state_space), -1.0e6, True

        # Update environment (location, travel time)
        travel_time = self.state.dist_matrix[self.state.truck_state.location][next_stop] / TRUCK_SPEED
        # travel_time += delay[next_stop] if needed, for unloading time. It might not be that useful though
        self.elapsed_time += travel_time
        self.route_log.append(next_stop)
        self.stops_visited += 1

        # Get truck temps (from temperature sensors)
        self.state.truck_state.temp = GET_TRUCK_TEMP  # TODO: Get truck temperature log from sensors, then use this to update shelf life.

        # Update agent state (produce shelf lifes, location)
        self.state.truck_state.location = next_stop
        self.state.visited[next_stop] = 1
        for p in self.state.produce:
            p.shelf_life_update(self.state.truck_state.temp, travel_time)

        # Calculate reward
        reward = self.calculate_reward(next_stop, travel_time)

        # Check to see if we have delivered everything (visited all destinations)
        done = self.check_done()
        if done:
            reward += self.return_to_warehouse()

        return self.state.get_state(), reward, done

    def legal_actions(self) -> list[int]:
        """Stops that can be visited from the current state."""
        return [stop for stop in range(1, self.state.num_stops) if self.is_legal(stop)]

    def return_to_warehouse(self) -> None:
        """Move back to the warehouse and add travel time. Return reward/penlaty for driving back."""
        travel_time = self.state.dist_matrix[self.state.truck_state.location][0] / TRUCK_SPEED
        self.elapsed_time += travel_time
        self.route_log.append(0)

        # No need to update temps/shelf life, truck is now empty

        return self.calculate_reward(0, travel_time)

    def calculate_reward(self, stop: int, travel_time: float) -> float:
        """Calculate reward with separate penalties for travel and delivery rewards."""
        # Reward for delivering produce (includes shelf life, big weight)
        reward = 0.0
        for p in self.state.produce:
            if p.destination == stop:
                reward += p.quantity * (p.shelf_life - p.shelf_life_requirement)  # Positive if on time, negative if late
        # Penalty for travel time (small weight)
        reward -= travel_time * TRAVEL_TIME_PENALTY_WEIGHT
        return reward

    def is_legal(self, stop: int) -> bool:
        """Can only visit all stops once, and can only go back to warehouse after visiting all other stops."""
        return stop > 0 and stop < self.state.num_stops and self.state.visited[stop] == 0

    def check_done(self) -> bool:
        """Determine if the episode should terminate."""
        return self.stops_visited == self.state.num_stops


# =========================== DQN AGENT =========================== #
DQN_MODEL_FILENAME = "dqn_agent.pth"

class DQNAgent:
    """
    Decision making agent. The only decision it gets to make is where to go next.
    Assume no control over the truck temperature.
    """
    def __init__(self, state_size: int, action_size: int, model: nn.Module = None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.tau = 0.005
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
            nn.Linear(self.state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

    def act(self, state: torch.Tensor, legal_actions: list[int] | None = None, use_epsilon: bool = True):
        if legal_actions is None:
            legal_actions = self.legal_actions_from_state(state)
        if not legal_actions:
            return 0

        if use_epsilon and random.random() <= self.epsilon:
            return random.choice(legal_actions)
        state = state.unsqueeze(0).to(self.device)
        q_values = self.model(state).squeeze(0)

        # Use a mask to automatically prohibit the agent from choosing any illegal stops.
        mask = torch.full((self.action_size,), float("-inf"), device=self.device)
        mask[legal_actions] = 0.0
        return torch.argmax(q_values + mask).item()

    def legal_actions_from_state(self, state: torch.Tensor) -> list[int]:
        visited_start = self.action_size ** 2
        visited = state[visited_start : visited_start + self.action_size]
        return [stop for stop in range(1, self.action_size) if visited[stop].item() == 0]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Get Q values of the current state using the action we would have chosen with current policy model.
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Zero if current state already gets terminated, otherwise max legal Q value of next state.
        next_q_values = self.target_model(next_states).detach()
        legal_mask = torch.full_like(next_q_values, float("-inf"))
        for row, next_state in enumerate(next_states):
            legal_actions = self.legal_actions_from_state(next_state.cpu())
            if legal_actions:
                legal_mask[row, legal_actions] = 0.0
        max_next_q_values = (next_q_values + legal_mask).max(1)[0]
        max_next_q_values = torch.where(torch.isfinite(max_next_q_values), max_next_q_values, torch.zeros_like(max_next_q_values))
        target_q_values = rewards + (~dones) * self.gamma * max_next_q_values

        loss: torch.Tensor = nn.SmoothL1Loss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def soft_update_target_model(self):
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_model.load_state_dict(target_net_state_dict)
    
    def save_model(self, file_path: str = DQN_MODEL_FILENAME):
        """Save the trained model to a file."""
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path: str = DQN_MODEL_FILENAME):
        """Load a trained model from a file."""
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            self.update_target_model()
            print(f"Model loaded from {file_path}")
        else:
            print(f"Model file {file_path} does not exist.")


# =========================== TRAINING LOOP =========================== #
def train_dqn(train_env: DeliveryEnv, initial_model: nn.Module = None, episodes=500, batch_size=64, pre_episodes=500):
    agent = DQNAgent(train_env.state.state_space, train_env.state.num_stops, initial_model)

    # PRE-TRAINING: Give the agent an example (the optimal route)
    # for pre_episode in range(pre_episodes):
    #     state = train_env.reset()
    #     total_reward = 0
    #     for truck_state in train_env.optimal_route:
    #         action = agent.act(state)
    #         next_state, reward, done = train_env.step(truck_state.location)
    #         agent.memory.append((state, truck_state.location, reward, next_state, done))
    #         state = next_state
    #         total_reward += reward

    #         if len(agent.memory) >= batch_size:
    #             agent.replay(batch_size)

    #         # Soft update target network weights
    #         agent.soft_update_target_model()
    #     assert done
    # agent.update_target_model()
    # print("Pretraining Total Reward:", total_reward)

    rewards = []        # Track rewards for visualization
    elapsed_times = []  # Track elapsed times
    losses = []         # Track loss over time

    for episode in range(episodes):
        state = train_env.reset()
        done = False
        total_reward = 0  # Total reward of the route.

        while not done:
            action = agent.act(state, train_env.legal_actions())
            next_state, reward, done = train_env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(agent.memory) >= batch_size:
                losses.append(agent.replay(batch_size))

            # Soft update target network weights
            agent.soft_update_target_model()

        # Decay per episode instead, since each episode is capped at n iterations.
        agent.decay_epsilon()

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        rewards.append(total_reward)
        elapsed_times.append(train_env.elapsed_time)  # Total time taken of the route.

    return agent, rewards, elapsed_times, losses

def train_dqn_agent(train_env: DeliveryEnv, episodes=500, batch_size=64, save_path=DQN_MODEL_FILENAME):
    # TODO: Should be trained on the same map, but also with different delays/uncertainties
    # Do this by placing the below in a loop, while keeping the model. So, agent can learn how to navigate many different possible delays

    # Initialize environment
    train_env.reset()

    # Train DQN agent
    print("Starting DQN training...")
    agent, rewards, elapsed_times, losses = train_dqn(train_env, None, episodes, batch_size)

    # Plot reward graph over training
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(rewards)
    ax2.plot(elapsed_times)
    ax3.plot(losses)
    plt.show()

    # Save the trained model
    agent.save_model(save_path)

    # ================= PRINT BEST ROUTE OF TRAINING DATASET ================= #
    print("\n--- Best Route Found In Training Dataset ---")
    train_env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(train_env.state.get_state(), train_env.legal_actions(), use_epsilon=False)
        _, reward, done = train_env.step(action)
        total_reward += reward

    # Display route log
    for log in train_env.route_log:
        print(log)
    print(f"Total time of route: {train_env.elapsed_time} h")
    print(f"Total reward of route: {total_reward}")

def run_model(test_env: DeliveryEnv, save_path: str = DQN_MODEL_FILENAME):
    """Load previously trained RL agent and run it to see what path it would choose."""
    agent = DQNAgent(test_env.state.state_space, test_env.state.num_stops)
    agent.load_model(save_path)

    print("\n--- Best Route Found ---")
    test_env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(test_env.state.get_state(), test_env.legal_actions(), use_epsilon=False)
        _, reward, done = test_env.step(action)
        total_reward += reward

    # Display route log
    for log in test_env.route_log:
        print(log)
    print(f"Total time of route: {test_env.elapsed_time} h")
    print(f"Total reward of route: {total_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save-path', default=DQN_MODEL_FILENAME, help="Path to save model to")
    parser.add_argument('-t', '--train', action='store_true')
    args = parser.parse_args()

    # Train the agent
    if args.train:
        train_env = DeliveryEnv(get_map(), get_produce(), [TruckState(i) for i in range(1, 11)])
        train_dqn_agent(train_env, episodes=100)

    # Use a real test dataset that's different than the one used in training (generates new variations from same base map)
    print("Testing previous model on test environment")
    test_env = DeliveryEnv(get_map(), get_produce(), [TruckState(i) for i in range(1, 11)])
    run_model(test_env)
