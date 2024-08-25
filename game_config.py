class GameConfig:
    # Game dimensions and basic settings
    WIDTH = 1450
    HEIGHT = 730
    NUM_SNAKES = 4
    SEGMENT_SIZE = 10
    INITIAL_FOOD = 250
    MAX_FOOD = 300
    MAX_FRAMES = 1000000
    FRAME_RATE = 1

    INPUT_SIZE = 27  
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 4

    # Actions
    ACTIONS = [
        (0, -1),  # Up
        (1, 0),   # Right
        (0, 1),   # Down
        (-1, 0)   # Left
    ]

    # Learning parameters
    BATCH_SIZE = 128
    MEMORY_SIZE = 2000
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.99995
    TARGET_UPDATE_FREQUENCY = 256*2

    # Training settings
    TRAIN_FREQUENCY = 2  # Ensure this is an integer
    CHECKPOINT_FREQUENCY = 1000

    # Snake colors
    SNAKE_COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128)   # Purple
    ]