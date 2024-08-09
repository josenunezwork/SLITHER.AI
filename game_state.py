from ai_snake import AISnake
from game_config import GameConfig
import random
from PyQt5.QtGui import QColor
from game_logic import GameLogic

class GameState:
    def __init__(self, shared_ai):
        self.shared_ai = shared_ai
        self.snakes = []
        self.food = []
        self.frame = 0
        self.alive_snakes = GameConfig.NUM_SNAKES
        self.snake_id_counter = 0
        self.reset()

    def reset(self):
        self.snakes = []
        start_positions = self._generate_start_positions()
        colors = self._generate_colors()
        
        for color, pos in zip(colors, start_positions):
            snake = AISnake(self.snake_id_counter, color, pos, GameConfig.SEGMENT_SIZE, GameConfig.WIDTH, GameConfig.HEIGHT, self.shared_ai)
            self.snakes.append(snake)
            self.snake_id_counter += 1        
        self.food = []
        self.spawn_food(GameConfig.INITIAL_FOOD)
        self.alive_snakes = GameConfig.NUM_SNAKES
        self.frame = 0

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
            empty_position = GameLogic.find_empty_position(GameConfig.WIDTH, GameConfig.HEIGHT, self.snakes)
            if empty_position:
                self.food.append(empty_position)
            else:
                print("Warning: Could not find an empty position to spawn food.")

    def update(self):
        self.frame += 1
        if len(self.food) < GameConfig.MAX_FOOD:
            self.spawn_food(4)
        self.alive_snakes = sum(1 for snake in self.snakes if snake.is_alive)
        
        for snake in self.snakes:
            if snake.is_alive:
                snake.update(self)  # Pass self (GameState) to the snake's update method

        # Check for collisions and food consumption
        collisions = GameLogic.check_collisions(self.snakes, self)
        eaten_food = GameLogic.check_food_consumption(self.snakes, self.food)
        self.food = [f for f in self.food if f not in eaten_food]

        # Spawn new food to replace eaten food
        self.spawn_food(len(eaten_food))

    def get_state(self):
        return {
            'snakes': [{'id': snake.id, 'segments': snake.segments, 'is_alive': snake.is_alive} for snake in self.snakes],
            'food': self.food,
            'frame': self.frame,
            'alive_snakes': self.alive_snakes
        }