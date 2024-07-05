import os
import random
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QColor, QPainter, QBrush
from PyQt5.QtCore import Qt, QTimer
from snake import Snake
from game_widget import GameWidget
from game_config import GameConfig
from game_state import GameState
from game_logic import GameLogic    
import torch
class SlitherIOGame(QWidget):
    def __init__(self):
        super().__init__()
        self.game_state = GameState()
        self.best_snake = None
        self.best_reward = float('-inf')
        self.initUI()



    def initUI(self):
        self.setGeometry(100, 100, GameConfig.WIDTH, GameConfig.HEIGHT + 50)
        self.setWindowTitle('Slither.io Game with Competitive Neural Networks')

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.game_widget = GameWidget(self.game_state)
        self.game_widget.setFixedSize(GameConfig.WIDTH, GameConfig.HEIGHT)
        layout.addWidget(self.game_widget)

        control_layout = QHBoxLayout()
        self.start_button = QPushButton('Start/Stop')
        self.start_button.clicked.connect(self.toggle_game)
        control_layout.addWidget(self.start_button)
        self.save_button = QPushButton('Save Best Snake')
        self.save_button.clicked.connect(self.save_best_snake)
        control_layout.addWidget(self.save_button)

        self.score_labels = [QLabel(f'Snake {i+1} Score: 0') for i in range(GameConfig.NUM_SNAKES)]
        for score_label in self.score_labels:
            control_layout.addWidget(score_label)

        layout.addLayout(control_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_game)

        self.show()


    def update_game(self):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for snake in self.game_state.snakes:
            if not snake.is_alive:
                snake.respawn_timer -= 1
                
                if snake.respawn_timer <= 0:
                    new_pos = GameLogic.find_empty_position(GameConfig.WIDTH, GameConfig.HEIGHT, self.game_state.snakes)
                    if new_pos:
                        snake.respawn(new_pos)
                        self.game_state.alive_snakes += 1
                        print(f"Snake {snake.id} respawned at {new_pos}. Alive snakes: {self.game_state.alive_snakes}")  # Debug print
                    else:
                        print(f"Failed to find empty position for Snake {snake.id}")  # Debug print
                continue

            old_state = snake.get_state([s for s in self.game_state.snakes if s != snake], self.game_state.food)
            action = snake.get_action(old_state)
            snake.change_direction(directions[action])
            snake.move()
            new_state = snake.get_state([s for s in self.game_state.snakes if s != snake], self.game_state.food)

            reward = snake.calculate_reward(action, old_state, new_state, self.game_state.food)
            snake.total_reward += reward

            snake.last_state = old_state
            snake.last_action = action

            snake.memory.add(old_state, action, reward, new_state, not snake.is_alive)

        
        eaten_food = GameLogic.check_food_consumption(self.game_state.snakes, self.game_state.food)
        GameLogic.check_collisions(self.game_state.snakes, self.game_state)

        for food in eaten_food:
            self.game_state.food.remove(food)

        if len(self.game_state.food) < GameConfig.MAX_FOOD:
            new_food = GameLogic.spawn_food(1, GameConfig.WIDTH, GameConfig.HEIGHT, self.game_state.snakes)
            self.game_state.food.extend(new_food)

        if self.game_state.games_played % 5 == 0:
            total_loss = 0
            for snake in self.game_state.snakes:
                loss = snake.train()
                if loss is not None:
                    total_loss += loss

            if self.game_state.games_played % 100 == 0:
                avg_loss = total_loss / len(self.game_state.snakes)
                print(f"Game {self.game_state.games_played}: Average Loss = {avg_loss:.4f}")

        self.game_state.update()

        for i, snake in enumerate(self.game_state.snakes):
            self.score_labels[i].setText(f'Snake {i+1} Score: {snake.length - 1} | Eps: {snake.epsilon:.2f} | Reward: {snake.total_reward:.1f}')

        self.game_widget.update()

        if self.game_state.games_played >= GameConfig.MAX_GAMES:
            print(f"Training completed after {self.game_state.games_played} games.")
            self.timer.stop()
               # Update the best snake
        for snake in self.game_state.snakes:
            if snake.total_reward > self.best_reward:
                self.best_reward = snake.total_reward
                self.best_snake = snake





    def toggle_game(self):
        if self.timer.isActive():
            self.timer.stop()
            self.save_best_snake()  # Save the best snake when stopping the game
        else:
            self.load_best_snake()  # Load the best snake when starting the game
            self.timer.start(7)

    def save_best_snake(self):
        if self.best_snake is not None:
            if not os.path.exists('saved_snakes'):
                os.makedirs('saved_snakes')
            torch.save({
                'dqn_state_dict': self.best_snake.dqn.state_dict(),
                'target_dqn_state_dict': self.best_snake.target_dqn.state_dict(),
                'optimizer_state_dict': self.best_snake.optimizer.state_dict(),
                'epsilon': self.best_snake.epsilon,
                'total_reward': self.best_snake.total_reward
            }, 'saved_snakes/best_snake.pth')
            print(f"Saved best snake with total reward: {self.best_reward}")

    def load_best_snake(self):
        if os.path.exists('saved_snakes/best_snake.pth'):
            checkpoint = torch.load('saved_snakes/best_snake.pth')
            for snake in self.game_state.snakes:
                snake.dqn.load_state_dict(checkpoint['dqn_state_dict'])
                snake.target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])
                snake.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                snake.epsilon = checkpoint['epsilon']
                snake.total_reward = checkpoint['total_reward']
            self.best_reward = checkpoint['total_reward']
            print(f"Loaded best snake with total reward: {self.best_reward}")
        else:
            print("No saved snake found. Starting with new snakes.")

