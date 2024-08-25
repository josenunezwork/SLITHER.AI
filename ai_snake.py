from snake import Snake
from game_config import GameConfig
from game_logic import GameLogic
from ai_controller import SnakeAI
import torch
import math

class AISnake(Snake):
    def __init__(self, id, color, start_pos, segment_size, game_width, game_height, game_state):
        super().__init__(id, color, start_pos, segment_size, game_width, game_height)
        self.ai = SnakeAI(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.last_state = None
        self.last_action = None
        self._total_reward = 0
        self.color_name = self.color_to_name(color)
        self.current_loss = 0
        self.current_epsilon = GameConfig.EPSILON_START
        self.game_state = game_state  # Store the GameState instance

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

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
        r, g, b = color
        return min(color_names.items(), key=lambda x: sum((a - b) ** 2 for a, b in zip((r, g, b), x[0])))[1]

    def get_action(self, state):
        return self.ai.get_action(state)

    def get_state(self, other_snakes, food):
        head_x, head_y = self.head
        state = []

        # Current direction
        direction_one_hot = [0, 0, 0, 0]
        direction_index = GameConfig.ACTIONS.index(self.direction)
        direction_one_hot[direction_index] = 1
        state.extend(direction_one_hot)

        # Enhanced food state
        food_state = self._get_enhanced_food_state(food, head_x, head_y)
        state.extend(food_state)

        # Get danger map
        danger_map = self._get_danger_map(other_snakes)
        state.extend(danger_map)

        # Add one-hot encoding of snake ID
        snake_id_encoding = [0] * GameConfig.NUM_SNAKES
        snake_id_encoding[self.id] = 1
        state.extend(snake_id_encoding)
        
        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def _get_enhanced_food_state(self, food, head_x, head_y, num_sectors=8):
        if not food:
            return [0] * (num_sectors + 3)  # 3 additional values for nearest food

        # Find the nearest food item
        nearest = min(food, key=lambda f: (f[0] - head_x)**2 + (f[1] - head_y)**2)

        # Nearest food relative position and distance
        dx, dy = nearest[0] - head_x, nearest[1] - head_y
        dist = math.sqrt(dx**2 + dy**2)
        rel_x, rel_y = dx / self.game_width, dy / self.game_height
        norm_dist = dist / math.sqrt(self.game_width**2 + self.game_height**2)

        # Food density map
        density_map = [0] * num_sectors
        for f in food:
            dx, dy = f[0] - head_x, f[1] - head_y
            angle = math.atan2(dy, dx)
            sector = int(((angle + math.pi) / (2 * math.pi) * num_sectors) % num_sectors)
            density_map[sector] += 1

        # Normalize density map
        max_density = max(density_map) if max(density_map) > 0 else 1
        density_map = [d / max_density for d in density_map]

        return [rel_x, rel_y, norm_dist] + density_map

    def _get_danger_map(self, other_snakes, num_sectors=8, max_distance=10):
        head_x, head_y = self.head
        danger_map = [1.0] * num_sectors  # Initialize with max safety

        def update_danger(segment_x, segment_y, is_self):
            dx, dy = segment_x - head_x, segment_y - head_y
            distance = math.sqrt(dx**2 + dy**2)
            if distance > max_distance * self.segment_size:
                return
            
            angle = math.atan2(dy, dx)
            sector = int(((angle + math.pi) / (2 * math.pi) * num_sectors) % num_sectors)
            
            # Normalize distance (closer = more dangerous)
            normalized_distance = 1 - (distance / (max_distance * self.segment_size))
            # Self-segments are slightly less dangerous
            danger_value = normalized_distance * (0.8 if is_self else 1.0)
            
            danger_map[sector] = min(danger_map[sector], 1 - danger_value)

        # Check own body
        for segment in self.segments[1:]:  # Exclude head
            update_danger(segment[0], segment[1], True)

        # Check other snakes
        for snake in other_snakes:
            if snake != self and snake.is_alive:
                for segment in snake.segments:
                    update_danger(segment[0], segment[1], False)

        return danger_map

    def update(self, other_snakes, food):
        if not self.is_alive:
            return

        current_state = self.get_state(other_snakes, food)
        action = self.get_action(current_state)
        
        new_direction = GameConfig.ACTIONS[action]
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction
        
        old_length = len(self.segments)
        self.move()
        ate_food = len(self.segments) > old_length
        
        collided = self._check_collision(other_snakes)
        
        if collided:
            self.die()
            print(f"{self.color_name} snake collided!")
        
        reward = self.calculate_reward(ate_food, collided, self.last_state, current_state, other_snakes, food)
        
        if self.last_state is not None:
            self.ai.update_memory(self.last_state, self.last_action, reward, current_state, collided)
        
        self.last_state = current_state
        self.last_action = action
        self._total_reward += reward
        self.epsilon = self.ai.epsilon

        # Call train method every TRAIN_FREQUENCY frames
        if self.game_state.frame % GameConfig.TRAIN_FREQUENCY == 0:
            
            loss, new_epsilon = self.train()
            if loss is not None:
                self.current_loss = loss
            self.current_epsilon = new_epsilon


    def calculate_reward(self, ate_food, collided, old_state, new_state, other_snakes, food):
        reward = 0

        if collided:
            return -1.0  # Severe penalty for collision

        if ate_food:
            reward += 1.0

        # Reward for getting closer to food
        if old_state is not None and new_state is not None:
            old_food_distance = math.sqrt(old_state[4]**2 + old_state[5]**2)
            new_food_distance = math.sqrt(new_state[4]**2 + new_state[5]**2)
            if new_food_distance < old_food_distance:
                reward += 0.1
            elif new_food_distance > old_food_distance:
                reward -= 0.05

        # Penalty for getting closer to other snakes or walls
        old_danger = sum(old_state[-8:]) / 8 if old_state is not None else 1
        new_danger = sum(new_state[-8:]) / 8
        danger_change = old_danger - new_danger
        reward += danger_change * 0.2

        return max(min(reward, 1), -1)  # Clamp reward between -1 and 1

    def _check_collision(self, other_snakes):
        return GameLogic.check_wall_collision(self) or any(
            GameLogic.check_body_collision(self, snake) for snake in other_snakes if snake != self and snake.is_alive)

    @property
    def total_reward(self):
        return self._total_reward

    def train(self):
        # Check if the training is enabled and the memory is not empty
        loss, new_epsilon = self.ai.train()
        if loss is not None:
            self.current_loss = loss
        self.current_epsilon = new_epsilon
        return loss, new_epsilon

    def move(self):
        new_head = (
            (self.head[0] + self.direction[0] * self.segment_size) % self.game_width,
            (self.head[1] + self.direction[1] * self.segment_size) % self.game_height
        )
        self.segments.insert(0, new_head)
        if len(self.segments) > self.length:
            self.segments.pop()

    def grow(self, amount=1):
        self.length += amount

    def die(self):
        self.is_alive = False
        self.respawn_timer = GameConfig.FRAME_RATE

    def respawn(self, new_pos):
        self.segments = [new_pos]
        self.direction = (1, 0)  # Reset direction to right
        self.is_alive = True
        self.length = 1
        self.respawn_timer = 0
        self.last_state = None
        self.last_action = None
        self._total_reward = 0