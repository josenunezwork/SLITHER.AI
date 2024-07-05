
import random
from game_config import GameConfig


class GameLogic:
    @staticmethod
    def check_collisions(snakes, game_state):
        for snake in snakes:
            if not snake.is_alive:
                continue

            # Check self-collision
            if snake.check_self_collision():
                GameLogic.kill_snake(snake, game_state)
                continue

            # Check collision with other snakes
            for other_snake in snakes:
                if other_snake != snake and other_snake.is_alive:
                    if snake.head == other_snake.head:  # Head-to-head collision
                        GameLogic.head_on_collision(snake, other_snake, game_state)
                    elif snake.head in other_snake.body[1:]:  # Head-to-body collision
                        GameLogic.body_collision(snake, other_snake, game_state)

    @staticmethod
    def head_on_collision(snake1, snake2, game_state):
        print(f"Head-on collision between Snake {snake1.id} and Snake {snake2.id}!")
        GameLogic.kill_snake(snake1, game_state)
        GameLogic.kill_snake(snake2, game_state)

    @staticmethod
    def body_collision(colliding_snake, hit_snake, game_state):
        print(f"Snake {colliding_snake.id} collided with Snake {hit_snake.id}'s body!")
        absorbed_length = colliding_snake.length // 2
        hit_snake.absorb(absorbed_length)
        kill_reward = 500 + (absorbed_length * 10)
        hit_snake.total_reward += kill_reward
        print(f"Snake {hit_snake.id} absorbed {absorbed_length} length and got a reward of {kill_reward}!")
        GameLogic.kill_snake(colliding_snake, game_state)

    @staticmethod
    def kill_snake(snake, game_state):
        snake.die()
        game_state.alive_snakes -= 1
        print(f"Snake {snake.id} died. Alive snakes: {game_state.alive_snakes}") 

    @staticmethod
    def check_food_consumption(snakes, food):
        eaten_food = []
        for snake in snakes:
            if not snake.is_alive:
                continue
            head = snake.body[0]
            eaten = [f for f in food if ((f[0] - head[0])**2 + (f[1] - head[1])**2)**0.5 < snake.segment_size]
            for f in eaten:
                snake.grow()
            eaten_food.extend(eaten)
        return eaten_food

    @staticmethod
    def spawn_food(count, width, height, snakes):
        new_food = []
        for _ in range(count):
            while True:
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                if not any((x, y) in snake.body for snake in snakes if snake.is_alive):
                    new_food.append((x, y))
                    break
        return new_food

    @staticmethod
    def find_empty_position(width, height, snakes):
        attempts = 0
        max_attempts = 100  # Limit the number of attempts to find an empty position
        while attempts < max_attempts:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if not any((x, y) in s.body for s in snakes if s.is_alive):
                return (x, y)
            attempts += 1
        print("Warning: Could not find an empty position after max attempts")  # Debug print
        return None 