import numpy as np


class PrioritizedMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.current_position = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if len(self.memory) > 0 else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.current_position] = (state, action, reward, next_state, done)

        self.priorities[self.current_position] = max_priority
        self.current_position = (self.current_position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[: len(self.memory)] ** self.alpha
        probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[index] for index in indices]
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for index, error in zip(indices, td_errors):
            self.priorities[index] = abs(error) + 1e-5

    def __len__(self):
        return len(self.memory)
