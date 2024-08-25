import sqlite3
import numpy as np

class MemoryDBHandler:
    def __init__(self, db_name='snake_memories.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snake_id INTEGER,
                state BLOB,
                action INTEGER,
                reward REAL,
                next_state BLOB,
                done INTEGER,
                priority REAL
            )
        ''')
        self.conn.commit()

    def save_memories(self, snake_id, memories):
        for memory in memories:
            state, action, reward, next_state, done, priority = (
                memory['state'], 
                memory['action'], 
                memory['reward'], 
                memory['next_state'], 
                memory['done'], 
                memory['priority']
            )

            state_bytes = np.array(state).tobytes()
            next_state_bytes = np.array(next_state).tobytes()

            self.cursor.execute('''
                INSERT INTO memories (snake_id, state, action, reward, next_state, done, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (snake_id, state_bytes, action, reward, next_state_bytes, int(done), float(priority)))

        self.conn.commit()
        print("Memories saved successfully.")

    def load_memories(self, snake_id=None):
        query = 'SELECT * FROM memories'
        params = []

        if snake_id is not None:
            query += ' WHERE snake_id = ?'
            params.append(snake_id)

        query += ' ORDER BY priority DESC LIMIT 4000'

        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        priorities = []

        for row in rows:
            try:
                state_data = np.frombuffer(row[2], dtype=np.float32).reshape(-1)
                next_state_data = np.frombuffer(row[5], dtype=np.float32).reshape(-1)

                actions.append(row[3])
                rewards.append(row[4])
                dones.append(bool(row[6]))
                priorities.append(float(row[7]))

                states.append(state_data)
                next_states.append(next_state_data)

            except ValueError as e:
                print(f"Error loading memory: {e} - Row data: {row}")
                continue

        print(f"Loaded {len(states)} memories from database.")
        return states, actions, rewards, next_states, dones, priorities

    def clear_memories(self, snake_id=None):
        if snake_id is None:
            self.cursor.execute('DELETE FROM memories')
        else:
            self.cursor.execute('DELETE FROM memories WHERE snake_id = ?', (snake_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()