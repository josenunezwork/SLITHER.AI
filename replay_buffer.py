import numpy as np
from collections import deque

class PrioritizedReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.memory = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.memory else 1.0
        self.memory.append((state, action, reward, next_state, done))
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities, replace=False)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = np.min([1., self.beta + self.beta_increment])

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + self.epsilon) ** self.alpha
    def __len__(self):
        return len(self.memory)
    def get_all_memories(self):
        return list(self.memory)
    def add_bulk(self, states, actions, rewards, next_states, dones, priorities=None):
        batch_size = len(states)
        if len(self.memory) + batch_size > self.capacity:
            overflow = len(self.memory) + batch_size - self.capacity
            self.memory = self.memory[overflow:]
            self.priorities = self.priorities[overflow:]
            self.position = 0
        
        if priorities is None:
            max_priority = np.max(self.priorities) if self.memory else 1.0
            priorities = np.full(batch_size, max_priority)
        
        for i in range(batch_size):
            self.memory.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))
            self.priorities[self.position] = priorities[i]
            self.position = (self.position + 1) % self.capacity