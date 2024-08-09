import sqlite3
import numpy as np
from game_config import GameConfig
class MemoryDBHandler:
    def __init__(self, db_name='snake_memories.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             snake_id INTEGER,
             state BLOB,
             action INTEGER,
             reward REAL,
             next_state BLOB,
             done INTEGER,
             priority REAL)
        ''')
        self.conn.commit()

    def save_memories(self, snake_id, memories):
        for state, action, reward, next_state, done, priority in memories:
            self.cursor.execute('''
                INSERT INTO memories (snake_id, state, action, reward, next_state, done, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (snake_id, state.tobytes(), action, reward, next_state.tobytes(), int(done), float(priority)))
        self.conn.commit()
    def load_memories(self, snake_id=None):
        if snake_id is None:
            self.cursor.execute('SELECT * FROM memories')
        else:
            self.cursor.execute('SELECT * FROM memories WHERE snake_id = ?', (snake_id,))
        rows = self.cursor.fetchall()
        
        states = np.zeros((len(rows), GameConfig.INPUT_SIZE), dtype=np.float32)
        next_states = np.zeros((len(rows), GameConfig.INPUT_SIZE), dtype=np.float32)
        actions = np.zeros(len(rows), dtype=np.int64)
        rewards = np.zeros(len(rows), dtype=np.float32)
        dones = np.zeros(len(rows), dtype=bool)
        priorities = np.zeros(len(rows), dtype=np.float32)

        for i, row in enumerate(rows):
            try:
                states[i] = np.frombuffer(row[2], dtype=np.float32)
                next_states[i] = np.frombuffer(row[5], dtype=np.float32)
                actions[i] = row[3]
                rewards[i] = row[4]
                dones[i] = bool(row[6])
                priorities[i] = float(row[7])
            except ValueError as e:
                print(f"Error loading memory: {e}")
                continue

        return states, actions, rewards, next_states, dones, priorities
    def clear_memories(self, snake_id=None):
        if snake_id is None:
            self.cursor.execute('DELETE FROM memories')
        else:
            self.cursor.execute('DELETE FROM memories WHERE snake_id = ?', (snake_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()