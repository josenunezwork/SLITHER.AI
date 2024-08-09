import sys
import os
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,QGroupBox, QFormLayout, QProgressBar
from PyQt5.QtCore import QTimer, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from memory_db_handler import MemoryDBHandler
from game_config import GameConfig
from dueling_dqn import DuelingDQN
from replay_buffer import PrioritizedReplay

class TrainingVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SARS Model Training Visualizer")
        self.setGeometry(100, 100, 1000, 800)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.setup_plot()
        self.setup_controls()
        self.setup_stats()

        self.memory_db = MemoryDBHandler()
        self.losses = []
        self.rewards = []
        self.training = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_training)

        self.model = DuelingDQN(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE).to(self.device)
        self.target_model = DuelingDQN(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE).to(self.device)
        self.memory = PrioritizedReplay(GameConfig.MEMORY_SIZE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=GameConfig.LEARNING_RATE, weight_decay=1e-5)
        self.epsilon = GameConfig.EPSILON_START

    def setup_plot(self):
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.loss_line, = self.ax.plot([], [], label='Loss')
        self.reward_line, = self.ax.plot([], [], label='Avg Reward')
        self.ax.set_xlabel('Training Iterations')
        self.ax.set_ylabel('Value')
        self.ax.set_title('SARS Model Training Progress')
        self.ax.legend()

    def setup_controls(self):
        controls_layout = QHBoxLayout()

        self.start_button = QPushButton('Start Training')
        self.start_button.clicked.connect(self.toggle_training)
        controls_layout.addWidget(self.start_button)

        self.save_button = QPushButton('Save and Stop')
        self.save_button.clicked.connect(self.save_and_stop)
        controls_layout.addWidget(self.save_button)

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_training)
        controls_layout.addWidget(self.reset_button)

        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(GameConfig.BATCH_SIZE)
        params_layout.addRow("Batch Size:", self.batch_size_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.00001, 1.0)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setValue(GameConfig.LEARNING_RATE)
        params_layout.addRow("Learning Rate:", self.learning_rate_spin)

        params_group.setLayout(params_layout)
        controls_layout.addWidget(params_group)

        self.layout.addLayout(controls_layout)

    def setup_stats(self):
        stats_group = QGroupBox("Training Statistics")
        stats_layout = QFormLayout()

        self.epsilon_label = QLabel('N/A')
        stats_layout.addRow("Epsilon:", self.epsilon_label)

        self.avg_reward_label = QLabel('N/A')
        stats_layout.addRow("Avg Reward:", self.avg_reward_label)

        self.loss_label = QLabel('N/A')
        stats_layout.addRow("Current Loss:", self.loss_label)

        self.memory_size_label = QLabel('0')
        stats_layout.addRow("Memory Size:", self.memory_size_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        stats_layout.addRow("Training Progress:", self.progress_bar)

        stats_group.setLayout(stats_layout)
        self.layout.addWidget(stats_group)

    def load_model_and_memories(self):
        if os.path.exists('saved_snakes/best_snake.pth'):
            checkpoint = torch.load('saved_snakes/best_snake.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['dqn_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_dqn_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Loaded model to {self.device}")
        else:
            print("No saved model found. Starting with new model.")

        states, actions, rewards, next_states, dones, priorities = self.memory_db.load_memories()
        print(f"Loaded {len(states)} memories from database")

        for i in range(len(states)):
            try:
                state_tensor = torch.from_numpy(states[i]).float().to(self.device)
                next_state_tensor = torch.from_numpy(next_states[i]).float().to(self.device)
                self.memory.add(state_tensor, actions[i], rewards[i], next_state_tensor, dones[i])
            except Exception as e:
                print(f"Error processing memory at index {i}: {e}")

        print(f"Successfully loaded {len(self.memory)} memories into the replay buffer")
        self.memory_size_label.setText(str(len(self.memory)))

    def toggle_training(self):
        if not self.training:
            self.load_model_and_memories()
            self.training = True
            self.timer.start(100)  # Update every 100ms
            self.start_button.setText('Pause Training')
        else:
            self.training = False
            self.timer.stop()
            self.start_button.setText('Resume Training')

    def update_training(self):
        if self.training:
            loss, avg_reward = self.train_step()
            if loss is not None:
                self.losses.append(loss)
                self.rewards.append(avg_reward)
                self.update_plot()
                self.update_stats()

    def train_step(self):
        if len(self.memory) < self.batch_size_spin.value():
            return None, None

        batch, indices, weights = self.memory.sample(self.batch_size_spin.value())
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * GameConfig.GAMMA * next_q_values

        loss = (torch.nn.functional.smooth_l1_loss(q_values.squeeze(), expected_q_values, reduction='none') * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(GameConfig.EPSILON_END, self.epsilon * GameConfig.EPSILON_DECAY)

        if len(self.losses) % GameConfig.TARGET_UPDATE_FREQUENCY == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item(), rewards.mean().item()

    def update_plot(self):
        self.loss_line.set_data(range(len(self.losses)), self.losses)
        self.reward_line.set_data(range(len(self.rewards)), self.rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def update_stats(self):
        avg_reward = np.mean(self.rewards[-100:]) if self.rewards else 0
        self.epsilon_label.setText(f'{self.epsilon:.4f}')
        self.avg_reward_label.setText(f'{avg_reward:.2f}')
        self.loss_label.setText(f'{self.losses[-1]:.4f}' if self.losses else 'N/A')
        self.memory_size_label.setText(str(len(self.memory)))
        self.progress_bar.setValue(int((1 - self.epsilon) * 100))

    def save_and_stop(self):
        self.training = False
        self.timer.stop()
        self.start_button.setText('Start Training')

        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')

        torch.save({
            'dqn_state_dict': self.model.state_dict(),
            'target_dqn_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, 'saved_models/best_model.pth')

        print("Model saved. Training stopped.")

    def reset_training(self):
        self.training = False
        self.timer.stop()
        self.start_button.setText('Start Training')
        self.losses = []
        self.rewards = []
        self.epsilon = GameConfig.EPSILON_START
        self.model = DuelingDQN(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE).to(self.device)
        self.target_model = DuelingDQN(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate_spin.value(), weight_decay=1e-5)
        self.memory.clear()
        self.update_plot()
        self.update_stats()
        print("Training reset. All progress cleared.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    visualizer = TrainingVisualizer()
    visualizer.show()
    sys.exit(app.exec_())