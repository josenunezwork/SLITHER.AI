import random
import numpy as np
from game_config import GameConfig

class GameLogic:
    @staticmethod
    def check_collisions(snakes, game_state):
        collisions = []
        for snake in snakes:
            if not snake.is_alive:
                continue
            for other_snake in snakes:
                if other_snake != snake and other_snake.is_alive:
                    if GameLogic.check_head_collision(snake, other_snake):
                        collisions.append((snake, other_snake, "head"))
                    elif GameLogic.check_body_collision(snake, other_snake):
                        collisions.append((snake, other_snake, "body"))
        
        for snake, other_snake, collision_type in collisions:
            if collision_type == "head":
                GameLogic.head_on_collision(snake, other_snake, game_state)
            elif collision_type == "body":
                GameLogic.body_collision(snake, other_snake, game_state)
        
        return collisions

    @staticmethod
    def head_on_collision(snake1, snake2, game_state):
        GameLogic.kill_snake(snake1, game_state)
        GameLogic.kill_snake(snake2, game_state)

    @staticmethod
    def body_collision(colliding_snake, hit_snake, game_state):
        GameLogic.kill_snake(colliding_snake, game_state)

    @staticmethod
    def check_head_collision(snake1, snake2):
        return GameLogic.distance(snake1.head, snake2.head) < snake1.segment_size

    @staticmethod
    def check_body_collision(snake1, snake2):
        for segment in snake2.segments[1:]:  # Skip the head
            if GameLogic.distance(snake1.head, segment) < snake1.segment_size:
                return True
        return False

    @staticmethod
    def distance(point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    @staticmethod
    def kill_snake(snake, game_state):
        snake.die()
        game_state.alive_snakes -= 1

    @staticmethod
    def check_food_consumption(snakes, food):
        eaten_food = []
        for snake in snakes:
            if not snake.is_alive:
                continue
            head = snake.segments[0]
            eaten = [f for f in food if ((f[0] - head[0])**2 + (f[1] - head[1])**2)**0.5 < snake.segment_size]
            for f in eaten:
                snake.grow()
            eaten_food.extend(eaten)
        return eaten_food

    @staticmethod
    def spawn_food(count, width, height, snakes):
        new_food = []
        for _ in range(count):
            attempts = 0
            while attempts < 100:  # Limit attempts to prevent infinite loop
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                if not any((x, y) in snake.segments for snake in snakes if snake.is_alive):
                    new_food.append((x, y))
                    break
                attempts += 1
            if attempts == 100:
                print("Warning: Could not find empty position for food after 100 attempts")
        return new_food

    @staticmethod
    def find_empty_position(width, height, snakes):
        attempts = 0
        max_attempts = 100  # Limit the number of attempts to find an empty position
        while attempts < max_attempts:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if not any((x, y) in s.segments for s in snakes if s.is_alive):
                return (x, y)
            attempts += 1
        print("Warning: Could not find an empty position after max attempts")  # Debug print
        return None

    @staticmethod
    def calculate_reward(old_state, new_state, ate_food, died, collision_type=None):
        reward = 0

        if ate_food:
            reward += 10
        if died:
            reward -= 20

        # Check if snake moved closer to food
        old_food_distance = old_state[8]  # Index 8 is the normalized food distance in our new state representation
        new_food_distance = new_state[8]
        
        if new_food_distance < old_food_distance:
            reward += 1
        else:
            reward -= 0.5

        # Additional reward based on the danger situation
        old_danger = sum(old_state[4:8])  # Indices 4-7 represent danger in each direction
        new_danger = sum(new_state[4:8])
        if new_danger < old_danger:
            reward += 0.5
        elif new_danger > old_danger:
            reward -= 0.5

        return reward

    @staticmethod
    def get_closest_food_distance(state):
        # This method is no longer needed with our new state representation
        # The food distance is already calculated and stored in the state
        return state[8]  # Index 8 is the normalized food distance in our new state representation