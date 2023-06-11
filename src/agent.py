import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from prioritized_memory import PrioritizedMemory


class Agent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0,
                 exploration_decay=0.995, batch_size=64, memory_size=10000, target_update_frequency=100):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = PrioritizedMemory(memory_size)
        self.model = DuelingNetwork(state_size, action_size)
        self.target_model = DuelingNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
        self.target_update_frequency = target_update_frequency
        self.training_steps = 0
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(state))
            return np.argmax(q_values.numpy())

    def update_value_function(self):
        if len(self.memory) < self.batch_size:
            return

        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        next_target_q_values = self.target_model(next_states)
        max_next_q_values, max_next_actions = torch.max(next_q_values, dim=1)

        target_q_values = q_values.clone()
        target_q_values[np.arange(self.batch_size), actions] = rewards + (1 - dones) * self.discount_factor * \
                                                               next_target_q_values[np.arange(
                                                                   self.batch_size), max_next_actions].detach()

        loss = self.loss_function(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def balance_exploration_and_exploitation(self):
        self.exploration_rate *= self.exploration_decay

    def train(self, environment, episodes, termination_condition=None):
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
            self.scheduler.step()
            if termination_condition and termination_condition(episode, self):
                break
