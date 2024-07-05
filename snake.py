# snake.py

import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from dueling_dqn import DuelingDQN
from replay_buffer import PrioritizedReplay
from game_config import GameConfig

class Snake:
    def __init__(self, id, color, start_pos, input_size, hidden_size, output_size, game_width, game_height):
        self.id = id  # Add this line
        self.color = color
        self.body = [start_pos]
        self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.is_alive = True
        self.length = 1
        self.segment_size = GameConfig.SEGMENT_SIZE
        self.respawn_timer = 0
        self.game_width = game_width
        self.game_height = game_height

        self.init_nn(input_size, hidden_size, output_size)
        
        self.memory = PrioritizedReplay(100)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_interval = 1000
        self.update_counter = 0

        self.total_reward = 0
        self.last_episode_reward = 0
        self.episode_count = 0
        self.last_state = None
        self.last_action = None

    def init_nn(self, input_size, hidden_size, output_size):
        self.dqn = DuelingDQN(input_size, hidden_size, output_size)
        self.target_dqn = DuelingDQN(input_size, hidden_size, output_size)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.AdamW(self.dqn.parameters(), lr=0.001)

    @property
    def head(self):
        return self.body[0]

    def move(self):
        new_head = (
            (self.head[0] + self.direction[0] * self.segment_size) % self.game_width,
            (self.head[1] + self.direction[1] * self.segment_size) % self.game_height
        )
        self.body.insert(0, new_head)
        if len(self.body) > self.length:
            self.body.pop()
    def die(self):
        self.is_alive = False
        self.respawn_timer = GameConfig.RESPAWN_DELAY
        print(f"Snake {self.id} died. Respawn timer set to {self.respawn_timer}")  # Debug print

    def respawn(self, new_pos):
        self.body = [new_pos]
        self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.is_alive = True
        self.length = 1
        self.respawn_timer = 0
        print(f"Snake {self.id} respawned at {new_pos}")  # Debug print
    def absorb(self, length):
        self.grow(length)

    def grow(self, amount=1):
        self.length += amount

    def change_direction(self, new_direction):
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

    def check_self_collision(self):
        return self.head in self.body[1:]

    def check_collision_with(self, other_snake):
        return self.head in other_snake.body

    def get_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.dqn(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(4)

   # In snake.py, update the get_state method

    def get_state(self, other_snakes, food):
        head = self.body[0]
        
        # Base features
        base_features = [
            head[0] / self.game_width,
            head[1] / self.game_height,
            self.direction[0],
            self.direction[1],
            len(self.body) / 100
        ]
        
        # Food features
        food_relative_positions = [(f[0] - head[0], f[1] - head[1]) for f in food]
        food_distances = [((dx**2 + dy**2)**0.5) for dx, dy in food_relative_positions]
        closest_foods = sorted(zip(food_distances, food_relative_positions))[:3]
        closest_foods += [(0, (0, 0))] * (3 - len(closest_foods))
        food_features = []
        for distance, (rel_x, rel_y) in closest_foods:
            food_features.extend([
                distance / ((self.game_width**2 + self.game_height**2)**0.5),
                rel_x / self.game_width,
                rel_y / self.game_height
            ])
        
        # Wall distances
        wall_distances = [
            head[0] / self.game_width,
            (self.game_width - head[0]) / self.game_width,
            head[1] / self.game_height,
            (self.game_height - head[1]) / self.game_height
        ]
        
        # Other snake features
        other_snake_features = []
        for other_snake in other_snakes:
            if other_snake.is_alive:
                other_head = other_snake.body[0]
                other_snake_features.extend([
                    (other_head[0] - head[0]) / self.game_width,
                    (other_head[1] - head[1]) / self.game_height,
                    other_snake.direction[0],
                    other_snake.direction[1],
                    len(other_snake.body) / 100
                ])
            else:
                other_snake_features.extend([0, 0, 0, 0, 0])
        
        # Pad other_snake_features if necessary
        other_snake_features += [0] * (5 * (GameConfig.NUM_SNAKES - 1) - len(other_snake_features))
        
        state = base_features + food_features + wall_distances + other_snake_features
        return torch.tensor(state, dtype=torch.float).unsqueeze(0)
    def calculate_reward(self, action, old_state, new_state, food_positions):
        reward = 0
        
        # Living reward
        reward += 1
        
        # Food reward
        if self.length > len(self.body):
            reward += 50
        
        # Moving towards food reward
        old_food_distance = min(((f[0] - old_state[0][0])**2 + (f[1] - old_state[0][1])**2)**0.5 for f in food_positions)
        new_food_distance = min(((f[0] - new_state[0][0])**2 + (f[1] - new_state[0][1])**2)**0.5 for f in food_positions)
        if new_food_distance < old_food_distance:
            reward += 5
        else:
            reward -= 1
        
        # Movement variety reward
        if len(self.body) > 2:
            if self.body[0] != self.body[2]:
                reward += 0.5
        
        return reward

    def train(self):
        if len(self.memory.memory) < self.batch_size:
            return

        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        current_q_values = self.dqn(states).gather(1, actions)
        
        # Double DQN
        next_actions = self.dqn(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_dqn(next_states).gather(1, next_actions).squeeze(1)
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = (current_q_values.squeeze() - expected_q_values).abs().detach().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)

        loss = (F.mse_loss(current_q_values.squeeze(), expected_q_values, reduction='none') * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.update_target_interval == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        return loss.item()

