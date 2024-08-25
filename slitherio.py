from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer
import torch
import numpy as np
from memory_db_handler import MemoryDBHandler
from game_config import GameConfig
from game_state import GameState
from ai_controller import SnakeAI
from game_widget import GameWidget
from game_logic import GameLogic
import os
from ai_snake import AISnake

class SlitherIOGame(QWidget):
    def __init__(self):
        super().__init__()
        self.best_snake = None
        self.best_reward = float('-inf')
        self.checkpoint_frequency = GameConfig.CHECKPOINT_FREQUENCY
        self.max_saved_memories = GameConfig.MEMORY_SIZE
        self.checkpoint_dir = 'checkpoints'
        
        self.memory_db = MemoryDBHandler()
        self.game_state = GameState()
        
        self.initUI()
        self.load_memories()

    def initUI(self):
        # Increase the window height to accommodate labels
        self.setGeometry(100, 100, GameConfig.WIDTH, GameConfig.HEIGHT + 100)
        self.setWindowTitle('Slither.io Game with Competitive Neural Networks')
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.game_widget = GameWidget(self.game_state)
        self.game_widget.setFixedSize(GameConfig.WIDTH, GameConfig.HEIGHT)
        main_layout.addWidget(self.game_widget)
        control_layout = QHBoxLayout()
        self.start_button = QPushButton('Start/Stop')
        self.start_button.clicked.connect(self.toggle_game)
        control_layout.addWidget(self.start_button)
        self.save_button = QPushButton('Save Best Snake')
        self.save_button.clicked.connect(self.save_best_snake)
        control_layout.addWidget(self.save_button)
        self.save_memories_button = QPushButton('Save Memories')
        self.save_memories_button.clicked.connect(self.save_memories)
        control_layout.addWidget(self.save_memories_button)
        main_layout.addLayout(control_layout)
        # Create a grid layout for the score labels
        score_layout = QGridLayout()
        score_layout.setHorizontalSpacing(10)
        score_layout.setVerticalSpacing(5)
        self.score_labels = []

        # Use a smaller font for the labels
        label_font = QFont()
        label_font.setPointSize(8)

        for i, snake in enumerate(self.game_state.snakes):
            color_name = self.color_to_name(snake.color)
            label = QLabel(f'{color_name}: S:0 E:1.00 R:0.0')
            label.setFont(label_font)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.score_labels.append(label)
            row = i // 4  # 4 labels per row
            col = i % 4
            score_layout.addWidget(label, row, col)

        # Add the score layout to the main layout
        main_layout.addLayout(score_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_game)

        self.show()

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
        
        r, g, b = color  # Unpack the tuple directly
        min_distance = float('inf')
        closest_color = "Unknown"
        for rgb, name in color_names.items():
            distance = sum((a - b) ** 2 for a, b in zip((r, g, b), rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        return closest_color


    def handle_checkpoints(self):
        if self.game_state.frame % self.checkpoint_frequency == 0:
            self.save_best_snake()

    def update_ui(self):
        self.update_stats()
        self.game_widget.update()
    def update_game(self):
        self.handle_checkpoints()
        self.game_state.update()
        
        for snake in self.game_state.snakes:
            if snake.is_alive:
                snake.update(self.game_state.snakes, self.game_state.food)
            else:
                self.handle_respawn(snake)
        self.manage_food()
        if self.game_state.frame % GameConfig.TRAIN_FREQUENCY == 0:
            self.train_ai()
        self.update_stats()
        self.game_widget.update()

    def handle_respawn(self, snake):
        snake.respawn_timer -= 1
        if snake.respawn_timer <= 0:
            new_pos = GameLogic.find_empty_position(GameConfig.WIDTH, GameConfig.HEIGHT, self.game_state.snakes)
            if new_pos:
                snake.respawn(new_pos)
                self.game_state.alive_snakes += 1

    def manage_food(self):
        food_count = len(self.game_state.food)
        if food_count < GameConfig.INITIAL_FOOD:
            food_to_spawn = GameConfig.INITIAL_FOOD - food_count
            self.game_state.spawn_food(food_to_spawn)

    def train_ai(self):
        for snake in self.game_state.snakes:
            if isinstance(snake, AISnake):
                loss, epsilon = snake.train()
                if loss is not None:
                    snake.current_loss = loss
                snake.current_epsilon = epsilon

    def update_snakes(self, collision_dict):
        for snake in self.game_state.snakes:
            if not snake.is_alive:
                self.handle_respawn(snake)
            else:
                self.update_single_snake(snake, collision_dict)

    def respawn_snake(self, snake, new_pos):
        snake.respawn(new_pos) 
        self.game_state.alive_snakes += 1

    def toggle_game(self):
        if self.timer.isActive():
            self.timer.stop()
            self.save_best_snake()
        else:
            self.load_best_snake()
            self.load_memories()
            self.timer.start(GameConfig.FRAME_RATE)

    def update_stats(self):
        for i, snake in enumerate(self.game_state.snakes):
            if snake.is_alive and isinstance(snake, AISnake):
                label_text = f'{snake.color_name}: S:{len(snake.segments)} E:{snake.current_epsilon:.2f} L:{snake.current_loss:.4f} R:{snake.total_reward:.1f}'
                self.score_labels[i].setText(label_text)
            else:
                self.score_labels[i].setText(f'{snake.color_name}: Dead')        
        self.game_widget.update()

    def load_memories(self):
        for snake in self.game_state.snakes:
            if isinstance(snake, AISnake):
                memories = self.memory_db.load_memories(snake.id)
                if memories:
                    states, actions, rewards, next_states, dones, priorities = memories
                    snake.ai.memory.add_bulk(states, actions, rewards, next_states, dones, priorities)
                    print(f"Loaded {len(states)} memories for {snake.color_name} snake.")
                else:
                    print(f"No valid memories found for {snake.color_name} snake.")
    def save_memories(self):
        for snake in self.game_state.snakes:
            if isinstance(snake, AISnake):
                memories_to_save = snake.ai.prepare_memories_for_saving()
                self.memory_db.save_memories(snake.id, memories_to_save)
        print("Memories saved to database.")

    def save_best_snake(self):
        current_best = max(self.game_state.snakes, key=lambda s: s.total_reward)
        if current_best.total_reward > self.best_reward:
            self.best_snake = current_best
            self.best_reward = current_best.total_reward
            torch.save({
                'dqn_state_dict': current_best.ai.dqn.state_dict(),
                'target_dqn_state_dict': current_best.ai.target_dqn.state_dict(),
                'optimizer_state_dict': current_best.ai.optimizer.state_dict(),
                'epsilon': current_best.ai.epsilon,
                'total_reward': self.best_reward
            }, 'saved_snakes/best_snake.pth')
            
            print(f"Saved best snake ({current_best.color_name}) with total reward: {self.best_reward}")

    def load_best_snake(self):
        if os.path.exists('saved_snakes/best_snake.pth'):
            print("Loading best snake...")
            checkpoint = torch.load('saved_snakes/best_snake.pth')
            self.best_snake = AISnake(
                0, 
                (255, 0, 0), 
                (GameConfig.WIDTH // 2, GameConfig.HEIGHT // 2), 
                GameConfig.SEGMENT_SIZE, 
                GameConfig.WIDTH, 
                GameConfig.HEIGHT,
                self  # Pass the SlitherIOGame instance as the game_state argument
            )
            self.best_snake.ai.dqn.load_state_dict(checkpoint['dqn_state_dict'])
            self.best_snake.ai.target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])
            self.best_snake.ai.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_snake.ai.epsilon = checkpoint['epsilon']
            self.best_reward = checkpoint.get('total_reward', 0)

            # Print out the loaded parameters for verification
            print(f"Epsilon: {self.best_snake.ai.epsilon}")
            print(f"Total Reward: {self.best_reward}")
            print(f"Optimizer state: {self.best_snake.ai.optimizer.state_dict()}")
        else:
            print("No saved snake found. Starting with new AI.")

    def closeEvent(self, event):
        self.memory_db.close()
        super().closeEvent(event)