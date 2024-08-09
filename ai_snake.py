import torch
import numpy as np
from snake import Snake
from game_config import GameConfig

class AISnake(Snake):
    def __init__(self, id, color, start_pos, segment_size, game_width, game_height, shared_ai):
        super().__init__(id, color, start_pos, segment_size, game_width, game_height)
        self.shared_ai = shared_ai
        self.device = shared_ai.device
        self.last_state = None
        self.last_action = None
        self._total_reward = 0
        self.color_name = self.color_to_name(color)

    def color_to_name(self, color):
        color_names = {
            (255, 0, 0): "Red",
            (0, 255, 0): "Green",
            (0, 0, 255): "Blue",
            (255, 255, 0): "Yellow",
            (255, 0, 255): "Magenta",
            (0, 255, 255): "Cyan",
            (255, 165, 0): "Orange",
            (128, 0, 128): "Purple"
        }
        r, g, b = color.red(), color.green(), color.blue()
        return min(color_names.items(), key=lambda x: sum((a - b) ** 2 for a, b in zip((r, g, b), x[0])))[1]


    def get_state(self, other_snakes, food):
        head_x, head_y = self.head
        state = []

        # Closest food information
        closest_food = min(food, key=lambda f: ((f[0] - head_x)**2 + (f[1] - head_y)**2)**0.5) if food else None
        if closest_food:
            food_distance = ((closest_food[0] - head_x)**2 + (closest_food[1] - head_y)**2)**0.5
            food_direction = [
                (closest_food[0] - head_x) / self.game_width,
                (closest_food[1] - head_y) / self.game_height
            ]
        else:
            food_distance = 1.0
            food_direction = [0, 0]

        state.extend(food_direction)
        state.append(food_distance / max(self.game_width, self.game_height))  # Normalize distance

        # Closest other snake information
        closest_snake_segment = None
        closest_snake_distance = float('inf')
        for snake in other_snakes:
            if snake != self and snake.is_alive:
                for segment in snake.segments:
                    distance = ((segment[0] - head_x)**2 + (segment[1] - head_y)**2)**0.5
                    if distance < closest_snake_distance:
                        closest_snake_distance = distance
                        closest_snake_segment = segment

        if closest_snake_segment:
            snake_distance = closest_snake_distance / max(self.game_width, self.game_height)
            snake_direction = [
                (closest_snake_segment[0] - head_x) / self.game_width,
                (closest_snake_segment[1] - head_y) / self.game_height
            ]
        else:
            snake_distance = 1.0
            snake_direction = [0, 0]

        state.extend(snake_direction)
        state.append(snake_distance)

        state.append(len(self.segments) / (self.game_width * self.game_height))

        return torch.tensor(state, dtype=torch.float32).to(self.device)
    def calculate_reward(self, ate_food, died, other_snakes, food):
        reward = 0
        if ate_food:
            reward += 10
        if died:
            reward -= 20
        
        if self.last_state is not None:
            current_state = self.get_state(other_snakes, food)
            
            old_food_distance = self.last_state[2].item()
            new_food_distance = current_state[2].item()
            distance_change = old_food_distance - new_food_distance
            scaling_factor = 1 / (new_food_distance + 1e-6)
            
            if distance_change > 0:
                reward += 2 * distance_change * scaling_factor
                print(f"{self.color_name} snake closer to food. Reward: {2 * distance_change * scaling_factor:.2f}")
            else:
                reward -= abs(distance_change) * scaling_factor
                print(f"{self.color_name} snake further from food. Penalty: {abs(distance_change) * scaling_factor:.2f}")

            old_snake_distance = self.last_state[5].item()
            new_snake_distance = current_state[5].item()
            snake_distance_change = new_snake_distance - old_snake_distance
            snake_scaling_factor = np.exp(-new_snake_distance * 5)
            
            if snake_distance_change > 0:
                snake_reward = snake_distance_change * snake_scaling_factor * 3
                reward += snake_reward
                print(f"{self.color_name} snake further from other snakes. Reward: {snake_reward:.2f}")
            else:
                snake_penalty = abs(snake_distance_change) * snake_scaling_factor * 5
                reward -= snake_penalty
                print(f"{self.color_name} snake closer to other snakes. Penalty: {snake_penalty:.2f}")

            old_length = self.last_state[6].item()
            new_length = current_state[6].item()
            
            if new_length > old_length:
                reward += 2
                print(f"{self.color_name} snake grew. Reward: 2")

        return reward

    def update(self, other_snakes, food):
        current_state = self.get_state(other_snakes, food)
        action = self.get_action(current_state)
        
        new_direction = GameConfig.ACTIONS[action]
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction
        
        old_length = len(self.segments)
        self.move()
        ate_food = len(self.segments) > old_length
        
        reward = self.calculate_reward(ate_food, not self.is_alive, other_snakes, food)
        
        if self.last_state is not None:
            self.shared_ai.update_memory(self.last_state, self.last_action, reward, current_state, not self.is_alive)
        
        self.last_state = current_state
        self.last_action = action

    def get_action(self, state):
        return self.shared_ai.get_action(state)

    def update_total_reward(self, reward):
        self._total_reward += reward

    @property
    def epsilon(self):
        return self.shared_ai.epsilon

    @property
    def total_reward(self):
        return self._total_reward

    def train(self):
        return self.shared_ai.train()