from game_config import GameConfig
from game_logic import GameLogic
import random
from PyQt5.QtGui import QColor
from ai_snake import AISnake

class GameState:
    def __init__(self):
        self.snakes = []
        self.food = []
        self.frame = 0
        self.alive_snakes = GameConfig.NUM_SNAKES
        self.snake_id_counter = 0
        self.reset()

    def reset(self):
        self.snakes = [
            AISnake(
                id=i,
                color=GameConfig.SNAKE_COLORS[i],
                start_pos=self.get_random_position(),
                segment_size=GameConfig.SEGMENT_SIZE,
                game_width=GameConfig.WIDTH,
                game_height=GameConfig.HEIGHT,
                game_state=self  # Pass the GameState instance
            ) for i in range(GameConfig.NUM_SNAKES)
        ]
        self.food = [self.get_random_position() for _ in range(GameConfig.INITIAL_FOOD)]
        self.frame = 0
        self.alive_snakes = GameConfig.NUM_SNAKES

    def get_random_position(self):
        return (
            random.randint(0, GameConfig.WIDTH - GameConfig.SEGMENT_SIZE),
            random.randint(0, GameConfig.HEIGHT - GameConfig.SEGMENT_SIZE)
        )

    def spawn_food(self, count):
        for _ in range(count):
            empty_position = GameLogic.find_empty_position(GameConfig.WIDTH, GameConfig.HEIGHT, self.snakes)
            if empty_position:
                self.food.append(empty_position)
            else:
                print("Warning: Could not find an empty position to spawn food.")

    def update(self):
        self.frame += 1
        if self.frame % GameConfig.TRAIN_FREQUENCY == 0:
            print(f"Frame {self.frame}: Updating game state")
        self.manage_food()
        self.alive_snakes = sum(1 for snake in self.snakes if snake.is_alive)
        
        for snake in self.snakes:
            if snake.is_alive:
                snake.update(self.snakes, self.food)
                if self.check_food_consumption(snake):
                    snake.grow()
                    self.spawn_food(1)  # Spawn new food to replace the eaten one
                if self.frame % GameConfig.TRAIN_FREQUENCY == 0:
                    print(f"Frame {self.frame}: Training snake {snake.id}")
                    snake.train()

        self.handle_collisions()

    def manage_food(self):
        if len(self.food) < GameConfig.MAX_FOOD:
            self.spawn_food(GameConfig.MAX_FOOD - len(self.food))

    def check_food_consumption(self, snake):
        head = snake.head
        eaten_food = [f for f in self.food if GameLogic.distance(f, head) < snake.segment_size]
        if eaten_food:
            self.food = [f for f in self.food if f not in eaten_food]
            snake.grow()
            return True
        return False

    def handle_collisions(self):
        collisions = GameLogic.check_collisions(self.snakes)
        for snake, other_snake, collision_type in collisions:
            if collision_type == "head":
                GameLogic.head_on_collision(snake, other_snake, self)
            elif collision_type == "body":
                GameLogic.body_collision(snake, other_snake, self)


