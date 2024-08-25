import numpy as np
from collections import deque
from game_config import GameConfig

class PrioritizedReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=GameConfig.EPSILON_START):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.memory = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def add(self, state, action, reward, next_state, done, priority):
        max_priority = np.max(self.priorities) if self.memory else 1.0
        self.memory.append((state, action, reward, next_state, done, priority))
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = np.array([priority for (_, _, _, _, _, priority) in self.memory], dtype=np.float32)
        # Normalize priorities to create a probability distribution
        if np.sum(priorities) == 0:
            probabilities = np.ones_like(priorities) / len(priorities)
        else:
            probabilities = priorities / np.sum(priorities)

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities, replace=False)
        samples = [self.memory[idx] for idx in indices]

        weights = (1 / len(self.memory) / probabilities[indices]) ** self.beta
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + self.epsilon) ** self.alpha
    
    def __len__(self):
        return len(self.memory)
    
    def get_all_memories(self):
        return list(self.memory)
    
    def add_bulk(self, states, actions, rewards, next_states, dones, priorities):
        for state, action, reward, next_state, done, priority in zip(states, actions, rewards, next_states, dones, priorities):
            priority = max(priority, 1e-5)
            self.memory.append((state, action, reward, next_state, done, priority))
