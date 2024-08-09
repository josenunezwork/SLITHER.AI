import os
import random
from collections import deque
import numpy as np
import torch
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout
from PyQt5.QtGui import QColor, QPainter, QBrush, QFont
from PyQt5.QtCore import Qt, QTimer
from snake import Snake
from game_widget import GameWidget
from game_config import GameConfig
from game_state import GameState
from game_logic import GameLogic
from ai_controller import SnakeAI
from memory_db_handler import MemoryDBHandler
from replay_buffer import PrioritizedReplay


class SlitherIOGame(QWidget):
    def __init__(self):
        super().__init__()
        self.best_snake = None
        self.best_reward = float('-inf')
        self.checkpoint_frequency = 1000
        self.max_saved_memories = 100000
        self.checkpoint_dir = 'checkpoints'
        self.memory_db = MemoryDBHandler()
        
        self.shared_ai = SnakeAI(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE)
        
        self.game_state = GameState(self.shared_ai)
        
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
        r, g, b = color.red(), color.green(), color.blue()
        min_distance = float('inf')
        closest_color = "Unknown"
        for rgb, name in color_names.items():
            distance = sum((a - b) ** 2 for a, b in zip((r, g, b), rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        return closest_color
    def update_game(self):
        directions = GameConfig.ACTIONS
        
        collisions = GameLogic.check_collisions(self.game_state.snakes, self.game_state)
        collision_dict = {snake: collision_type for snake, _, collision_type in collisions}
        
        self.update_labels()
        
        if self.game_state.frame % self.checkpoint_frequency == 0:
            self.save_best_snake()

        self.game_widget.update()
        
        eaten_food = []
        for snake in self.game_state.snakes:
            if not snake.is_alive:
                snake.respawn_timer -= 1
                if snake.respawn_timer <= 0:
                    new_pos = GameLogic.find_empty_position(GameConfig.WIDTH, GameConfig.HEIGHT, self.game_state.snakes)
                    if new_pos:
                        snake.respawn(new_pos)
                        self.game_state.alive_snakes += 1
                    else:
                        print(f"Failed to find empty position for Snake {snake.id}")
                continue

            other_snakes = [s for s in self.game_state.snakes if s != snake]
            snake.update(other_snakes, self.game_state.food)

            # Check for food consumption
            snake_eaten_food = GameLogic.check_food_consumption([snake], self.game_state.food)
            eaten_food.extend(snake_eaten_food)

        # Remove eaten food
        self.game_state.food = [f for f in self.game_state.food if f not in eaten_food]

        # Spawn new food to replace eaten food and maintain minimum food count
        food_to_spawn = max(len(eaten_food), GameConfig.INITIAL_FOOD - len(self.game_state.food))
        self.game_state.spawn_food(food_to_spawn)

        # Train the shared AI periodically
        if self.game_state.frame % 5 == 0:
            loss = self.shared_ai.train()
            if loss is not None:
                print(f"Training loss: {loss}")

        self.game_state.frame += 1


   

    def toggle_game(self):
        if self.timer.isActive():
            self.timer.stop()
            self.save_best_snake()  # Save the best snake when stopping the game
        else:
            self.load_best_snake()  # Load the best snake when starting the game
            self.load_memories()
            self.timer.start(GameConfig.FRAME_RATE)

    def update_labels(self):
        for i, snake in enumerate(self.game_state.snakes):
            if snake.is_alive:
                label_text = f'{snake.color_name}: S:{len(snake.segments)} E:{snake.epsilon:.2f} R:{snake.total_reward:.1f}'
                self.score_labels[i].setText(label_text)
            else:
                self.score_labels[i].setText(f'{snake.color_name}: Dead')

    def save_best_snake(self):
        current_best = max(self.game_state.snakes, key=lambda s: s.total_reward)
        if current_best.total_reward > self.best_reward:
            self.best_snake = current_best
            self.best_reward = current_best.total_reward
            
            if not os.path.exists('saved_snakes'):
                os.makedirs('saved_snakes')
            
            torch.save({
                'dqn_state_dict': self.shared_ai.dqn.state_dict(),
                'target_dqn_state_dict': self.shared_ai.target_dqn.state_dict(),
                'optimizer_state_dict': self.shared_ai.optimizer.state_dict(),
                'epsilon': self.shared_ai.epsilon,
                'total_reward': self.best_reward
            }, 'saved_snakes/best_snake.pth')
            
            print(f"Saved best snake ({self.best_snake.color_name}) with total reward: {self.best_reward}")

    def load_best_snake(self):
        if os.path.exists('saved_snakes/best_snake.pth'):
            checkpoint = torch.load('saved_snakes/best_snake.pth')
            self.shared_ai.dqn.load_state_dict(checkpoint['dqn_state_dict'])
            self.shared_ai.target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])
            self.shared_ai.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.shared_ai.epsilon = checkpoint['epsilon']
        else:
            print("No saved snake found. Starting with new AI.")

    def ensure_checkpoint_directory(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print(f"Created checkpoint directory: {self.checkpoint_dir}")

    def save_memories(self):
        for snake in self.game_state.snakes:
            memories = self.shared_ai.get_all_memories()  # This should return memories with priorities
            self.memory_db.save_memories(snake.id, memories)
        print("Memories saved to database")

    def load_memories(self):
        memories = self.memory_db.load_memories()  # Load all memories
        states, actions, rewards, next_states, dones, priorities = [], [], [], [], [], []

        for memory in memories:
            try:
                if len(memory) == 6:
                    state, action, reward, next_state, done, priority = memory
                elif len(memory) == 5:
                    state, action, reward, next_state, done = memory
                    priority = 1.0
                else:
                    raise ValueError(f"Unexpected memory format: {memory}")
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                priorities.append(priority)
            except Exception as e:
                print(f"Error processing memory: {e}")

        if states:  # Check if we have any valid memories
            try:
                self.shared_ai.memory.add_bulk(
                    torch.tensor(np.array(states), dtype=torch.float32),
                    torch.tensor(np.array(actions), dtype=torch.long),
                    torch.tensor(np.array(rewards), dtype=torch.float32),
                    torch.tensor(np.array(next_states), dtype=torch.float32),
                    torch.tensor(np.array(dones), dtype=torch.bool),
                    np.array(priorities)
                )
                print(f"Loaded {len(states)} memories from database")
            except Exception as e:
                print(f"Error adding memories to replay buffer: {e}")
        else:
            print("No valid memories found in the database.")
    def prepare_memories_for_saving(self, memory_buffer):
        if len(memory_buffer) > self.max_saved_memories:
            probs = memory_buffer.priorities[:len(memory_buffer)]
            probs /= probs.sum()
            indices = np.random.choice(len(memory_buffer), self.max_saved_memories, p=probs, replace=False)
            memories_to_save = [memory_buffer.memory[i] for i in indices]
            priorities_to_save = memory_buffer.priorities[indices]
        else:
            memories_to_save = memory_buffer.memory
            priorities_to_save = memory_buffer.priorities[:len(memory_buffer)]

        return [{
            'state': state.cpu().numpy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu().numpy(),
            'done': done,
            'priority': float(priority)
        } for (state, action, reward, next_state, done), priority in zip(memories_to_save, priorities_to_save)]
    def closeEvent(self, event):
        self.memory_db.close()
        super().closeEvent(event)