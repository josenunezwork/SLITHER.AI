import torch
import torch.optim as optim
import random
from dueling_dqn import DuelingDQN
from replay_buffer import PrioritizedReplay
from game_config import GameConfig
import numpy as np

class SnakeAI:
    def __init__(self, input_size, hidden_size, output_size):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        self.dqn = DuelingDQN(input_size, hidden_size, output_size).to(self.device)
        self.target_dqn = DuelingDQN(input_size, hidden_size, output_size).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.AdamW(self.dqn.parameters(), lr=GameConfig.LEARNING_RATE, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1000, verbose=True)
        self.best_reward = float('-inf')
        
        self.memory = PrioritizedReplay(GameConfig.MEMORY_SIZE)
        self.epsilon = GameConfig.EPSILON_START
        
        self.total_reward = 0.0
        self.update_counter = 0



    def update_memory(self, snake, other_snakes, food, action, ate_food, collided):
        old_state = snake.get_state(other_snakes, food)
        new_state = snake.get_state(other_snakes, food) if not collided else None
        reward = snake.calculate_reward(ate_food, collided, old_state, new_state, other_snakes, food)

        state_tensor = old_state.to(self.device)
        next_state_tensor = new_state.to(self.device) if new_state is not None else None
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)

        max_priority = np.max(self.memory.priorities) if len(self.memory) > 0 else 1.0
        
        self.memory.add(state_tensor, action, reward_tensor.item(), next_state_tensor, collided, max_priority)
        self.total_reward += reward_tensor.item()
        snake._total_reward += reward_tensor.item()


    def train(self, num_iterations=1):
        if len(self.memory) < GameConfig.BATCH_SIZE:
            print("Not enough memories to train")
            return None, self.epsilon
        
        total_loss = 0
        for _ in range(num_iterations):
            batch, indices, weights = self.memory.sample(GameConfig.BATCH_SIZE)
            
            states, actions, rewards, next_states, dones, _ = zip(*batch)
            
            states = torch.stack([s.to(self.device) for s in states])
            actions = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            next_states = torch.stack([s.to(self.device) for s in next_states])
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

            current_q_values = self.dqn(states).gather(1, actions)

            with torch.no_grad():
                next_q_values = self.target_dqn(next_states).max(1)[0]
                expected_q_values = rewards + (1 - dones) * GameConfig.GAMMA * next_q_values

            td_errors = (current_q_values.squeeze() - expected_q_values).abs().detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

            loss = (torch.nn.functional.smooth_l1_loss(current_q_values.squeeze(), expected_q_values, reduction='none') * weights).mean()
            

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            self.update_counter += 1
            if self.update_counter % GameConfig.TARGET_UPDATE_FREQUENCY == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.epsilon = max(GameConfig.EPSILON_END, self.epsilon * GameConfig.EPSILON_DECAY)
        avg_reward = self.total_reward / (self.update_counter + 1)
        self.scheduler.step(avg_reward)

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            # Increase training frequency if performance improves
            GameConfig.TRAIN_FREQUENCY = max(1, GameConfig.TRAIN_FREQUENCY - 1)
        else:
            # Decrease training frequency if performance stagnates
            GameConfig.TRAIN_FREQUENCY = min(10, GameConfig.TRAIN_FREQUENCY + 1)
        return total_loss / num_iterations, self.epsilon


    def get_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = state.to(self.device)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                q_values = self.dqn(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(GameConfig.OUTPUT_SIZE)

    def update_memory(self, state, action, reward, next_state, done):
        if state is None or next_state is None:
            return  # Skip updating memory if either state is None

        state_tensor = state.to(self.device)
        next_state_tensor = next_state.to(self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)

        max_priority = np.max(self.memory.priorities) if len(self.memory) > 0 else 1.0
        
        self.memory.add(state_tensor, action, reward_tensor.item(), next_state_tensor, done, max_priority)
        self.total_reward += reward_tensor.item()

    
    def get_all_memories(self):
        return [(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done, priority) 
                for (state, action, reward, next_state, done), priority in zip(self.memory.memory, self.memory.priorities[:len(self.memory.memory)])]

    def add_bulk_memories(self, memories):
        for state, action, reward, next_state, done, priority in memories:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            self.memory.add(state_tensor, action, reward, next_state_tensor, done, priority)  # Pass six arguments