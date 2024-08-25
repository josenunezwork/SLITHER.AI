import random
from game_config import GameConfig

class GameLogic:
    @staticmethod    
    def check_collisions(snakes):
        collisions = []
        for i, snake in enumerate(snakes):
            if not snake.is_alive:
                continue
            # Check collision with walls
            if GameLogic.check_wall_collision(snake):
                collisions.append((snake, None, "wall"))
            # Check collision with other snakes
            for j, other_snake in enumerate(snakes):
                if i != j and other_snake.is_alive:
                    if GameLogic.check_head_collision(snake, other_snake):
                        collisions.append((snake, other_snake, "head"))
                    elif GameLogic.check_body_collision(snake, other_snake):
                        collisions.append((snake, other_snake, "body"))
        return collisions

    @staticmethod
    def check_wall_collision(snake):
        head_x, head_y = snake.head
        return (head_x < 0 or head_x >= GameConfig.WIDTH or 
                head_y < 0 or head_y >= GameConfig.HEIGHT)

    @staticmethod
    def head_on_collision(snake1, snake2, game_state):
        snake1.die()
        snake2.die()
        game_state.alive_snakes -= 2

    @staticmethod
    def body_collision(colliding_snake, hit_snake, game_state):
        colliding_snake.die()
        game_state.alive_snakes -= 1

    @staticmethod
    def check_head_collision(snake1, snake2):
        return GameLogic.distance(snake1.head, snake2.head) < snake1.segment_size

    @staticmethod
    def check_body_collision(snake1, snake2):
        return any(GameLogic.distance(snake1.head, segment) < snake1.segment_size 
                   for segment in snake2.segments[1:])

    @staticmethod
    def distance(point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    @staticmethod
    def find_empty_position(width, height, snakes):
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if not any(GameLogic.distance((x, y), segment) < snake.segment_size 
                       for snake in snakes if snake.is_alive
                       for segment in snake.segments):
                return (x, y)
            attempts += 1
        print("Warning: Could not find an empty position after max attempts")
        return None