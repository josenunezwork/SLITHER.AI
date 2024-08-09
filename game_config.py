class GameConfig:
    # Game dimensions and basic settings
    WIDTH = 1300
    HEIGHT = 700
    NUM_SNAKES = 4
    SEGMENT_SIZE = 10
    INITIAL_FOOD = 100
    MAX_FOOD = 410
    MAX_FRAMES = 10000000
    FRAME_RATE = 1

    # Snake vision and state
    INPUT_SIZE =7  
    HIDDEN_SIZE = 1000  
    OUTPUT_SIZE = 4

    # Actions
    ACTIONS = [
        (0, -1),  # Up
        (1, 0),   # Right
        (0, 1),   # Down
        (-1, 0)   # Left
    ]

    # Learning parameters
    BATCH_SIZE = 512  
    MEMORY_SIZE = 10000
    LEARNING_RATE = 0.001
    GAMMA = 0.2
    EPSILON_START = 1.0
    EPSILON_END = 0.2
    EPSILON_DECAY = 0.9995
    TARGET_UPDATE_FREQUENCY = 100

    # Training settings
    TRAIN_FREQUENCY = 5
    CHECKPOINT_FREQUENCY = 1000