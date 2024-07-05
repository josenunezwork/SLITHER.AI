from game_config import GameConfig
from snake import Snake
import random
from PyQt5.QtGui import QColor

class GameState:
    def __init__(self):
        self.snakes = []
        self.food = []
        self.games_played = 0
        self.alive_snakes = GameConfig.NUM_SNAKES
        self.best_snake = None
        self.best_reward = float('-inf')
        self.snake_id_counter = 0  # Add this line
        self.reset()

    def reset(self):
        self.snakes = []
        start_positions = self._generate_start_positions()
        colors = self._generate_colors()
        
        for color, pos in zip(colors, start_positions):
            snake = Snake(self.snake_id_counter, color, pos, GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, 
                          GameConfig.OUTPUT_SIZE, GameConfig.WIDTH, GameConfig.HEIGHT)
            self.snakes.append(snake)
            self.snake_id_counter += 1  # Increment the counter        
        self.food = []
        self.spawn_food(GameConfig.INITIAL_FOOD)
        self.alive_snakes = GameConfig.NUM_SNAKES
        self.games_played = 0

    def _generate_start_positions(self):
        return [(100, 300), (700, 300)] + [
            (random.randint(0, GameConfig.WIDTH), random.randint(0, GameConfig.HEIGHT))
            for _ in range(GameConfig.NUM_SNAKES - 2)
        ]

    def _generate_colors(self):
        return [QColor(255, 0, 0), QColor(0, 0, 255)] + [
            QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(GameConfig.NUM_SNAKES - 2)
        ]

    def spawn_food(self, count):
        for _ in range(count):
            while True:
                x = random.randint(0, GameConfig.WIDTH - 1)
                y = random.randint(0, GameConfig.HEIGHT - 1)
                if not any((x, y) in snake.body for snake in self.snakes if snake.is_alive):
                    self.food.append((x, y))
                    break

    def update(self):
        self.games_played += 1
        if len(self.food) < GameConfig.MAX_FOOD:
            self.spawn_food(1)
        self.alive_snakes = sum(1 for snake in self.snakes if snake.is_alive)
