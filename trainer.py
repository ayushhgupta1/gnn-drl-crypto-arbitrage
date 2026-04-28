import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
from collections import deque

from env import ArbitrageEnv
from data_feed import DataFeed
from model import GNNArbitrageAgent


class ReplayBuffer:
    def __init__(self, capacity=20000):  # Increased capacity for the larger state space
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Trainer:
    def __init__(self):
        self.feed = DataFeed(mode="mock")
        self.env = ArbitrageEnv(self.feed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # INIT UPDATE: 9 Nodes (3 assets * 3 exchanges), 19 Actions
        self.policy_net = GNNArbitrageAgent(num_nodes=9, node_features=5, action_dim=19).to(self.device)
        self.target_net = GNNArbitrageAgent(num_nodes=9, node_features=5, action_dim=19).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.memory = ReplayBuffer(capacity=20000)

        self.batch_size = 64
        self.gamma = 0.99

        # Slower epsilon decay because 19 actions require much more exploration than 4
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9997
        self.target_update_freq = 20

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(state).gather(1, action).squeeze(1)

        # Max Q(s', a') from Target Network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state).max(1)[0]
            expected_q_values = reward + (self.gamma * max_next_q_values * (1 - done))

        # Huber Loss for stability against large reward spikes
        loss = nn.SmoothL1Loss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping to prevent explosion in the Spatial routing
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def train(self, episodes=10000):
        # Create fresh CSV log header
        with open("spatial_training_logs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Step", "Action_Routed", "Portfolio_Value", "Reward", "Epsilon"])

        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)

                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.optimize_model()

                step = self.env.current_step

                # Log Spatial Routes every 50 steps
                if step % 50 == 0 or done:
                    action_name = self.env.action_map[action]["name"]
                    with open("spatial_training_logs.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            ep + 1,
                            step,
                            action_name,
                            f"{self.env.portfolio_value_usd:.2f}",
                            f"{reward:.4f}",
                            f"{self.epsilon:.4f}"
                        ])

                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (ep + 1) % 10 == 0:
                print(f"Episode {ep + 1:4d} | PnL: {total_reward:7.2f} | Epsilon: {self.epsilon:.3f}")

            if (ep + 1) % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Save checkpoints as 'spatial' to avoid overwriting your single-exchange demo
            if (ep + 1) % 5000 == 0:
                torch.save(self.policy_net.state_dict(), f"gnn_spatial_model_{ep + 1}.pth")
                print(f"Auto-saved spatial model checkpoint at episode {ep + 1}")