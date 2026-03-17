"""
DQN Agent Implementation
Basic DQN with experience replay and target network
Not using double DQN or dueling - keeping it simple for the project
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class ReplayBuffer:
    """
    Simple replay buffer - maybe not the most efficient but works for now.
    Stores transitions and samples random batches for training.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)  # auto-discards old stuff

    def push(self, state, action, reward, next_state, done, next_valid_actions):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done, next_valid_actions))

    def sample(self, batch_size):
        """Random sample of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """
    Simple neural network for Q-value approximation.
    3 layers - not too deep since we want it to train reasonably fast.
    """

    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()

        # simple feedforward network
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        # using ReLU - pretty standard
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # no activation on output - raw Q values


class DQNAgent:
    """
    DQN Agent with epsilon-greedy exploration and experience replay.
    Based on the original DeepMind Atari paper but simplified.
    """

    def __init__(self, state_size, max_actions, learning_rate=0.001):
        self.state_size = state_size
        self.max_actions = max_actions  # max neighbors any node has

        # device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Q networks - main and target
        self.q_network = DQN(state_size, max_actions).to(self.device)
        self.target_network = DQN(state_size, max_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # target network doesn't train

        # optimizer - Adam is pretty standard
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity=50000)

        # hyperparameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # start fully random
        self.epsilon_min = 0.01  # still explore a tiny bit
        self.epsilon_decay = 0.995  # slowly reduce exploration

        self.batch_size = 64
        self.target_update_freq = 1000  # update target network every N steps
        self.train_step = 0

    def select_action(self, state, valid_actions, training=True):
        """
        Epsilon-greedy action selection.
        With probability epsilon, pick random action.
        Otherwise, pick action with highest Q-value.
        """
        if training and random.random() < self.epsilon:
            # random action (exploration)
            return random.randrange(valid_actions)

        # greedy action (exploitation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)

            # only consider valid actions (neighbors that exist)
            valid_q = q_values[0, :valid_actions]
            return valid_q.argmax().item()

    def store_transition(self, state, action, reward, next_state, done, next_valid_actions):
        """Store transition in replay buffer."""
        self.replay_buffer.push(
            state, action, reward, next_state, done, next_valid_actions
        )

    def train(self):
        """
        Train the Q-network using experience replay.
        Returns the loss value for logging.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None  # not enough samples yet

        # sample random batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, next_valid_actions_list = zip(*batch)

        # convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # current Q values
        current_q = self.q_network(states).gather(1, actions).squeeze()

        # next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)

            # mask invalid actions (set to very negative)
            for i, nva in enumerate(next_valid_actions_list):
                next_q_values[i, nva:] = -float('inf')

            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # compute loss (MSE)
        loss = nn.MSELoss()(current_q, target_q)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Reduce exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save model weights."""
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        """Load model weights."""
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())


if __name__ == "__main__":
    # quick test
    agent = DQNAgent(state_size=200, max_actions=10)  # small test
    print(f"Agent created with {sum(p.numel() for p in agent.q_network.parameters())} parameters")
