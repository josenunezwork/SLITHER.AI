# game_config.py

class GameConfig:
    WIDTH = 800
    HEIGHT = 600
    NUM_SNAKES = 4
    SEGMENT_SIZE = 10
    INITIAL_FOOD = 10
    MAX_FOOD = 20
    MAX_GAMES = 10000
    RESPAWN_DELAY = 50
    
    # Calculate INPUT_SIZE
    BASE_FEATURES = 5  # snake's own x, y, direction_x, direction_y, length
    FOOD_FEATURES = 9  # 3 closest food items, each with distance, relative_x, relative_y
    WALL_DISTANCES = 4  # distance to 4 walls
    OTHER_SNAKE_FEATURES = 5 * (NUM_SNAKES - 1)  # 5 features per other snake
    
    INPUT_SIZE = BASE_FEATURES + FOOD_FEATURES + WALL_DISTANCES + OTHER_SNAKE_FEATURES
    
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 4