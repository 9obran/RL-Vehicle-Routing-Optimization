import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class Agent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, batch_size=64, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.model, self.optimizer, self.loss_function = self.create_model()

    def create_model(self):
        """Create the neural network model, optimizer, and loss function."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_function = nn.MSELoss()
        return model, optimizer, loss_function

    def remember(self, state, action, reward, next_state, done):
        """Store the agent's experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """Select an action based on the current state."""
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(state))
            return np.argmax(q_values.numpy())

    def update_value_function(self):
        """Update the Q-value function using the agent's experiences."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        target_q_values = self.model(states)
        next_q_values = self.model(next_states)
        max_next_q_values, _ = torch.max(next_q_values, dim=1)

        target_q_values[np.arange(self.batch_size), actions] = rewards + (1 - dones) * self.discount_factor * max_next_q_values
        predicted_q_values = self.model(states)

        loss = self.loss_function(predicted_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def balance_exploration_and_exploitation(self):
        """Adjust the exploration rate over time."""
        self.exploration_rate *= self.exploration_decay

    def train(self, environment, episodes, termination_condition=None):
        """
        Train the agent using the given environment and number of episodes.
        Optionally, provide a termination_condition function to stop the training process.
        """
        for episode in range(episodes):
            state = environment.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done = environment.step(action)
                self.remember(state, action, reward, next_state, done)
                self.update_value_function()
                state = next_state

            self.balance_exploration_and_exploitation()
            if termination_condition and termination_condition(episode, self):
                break

