import sqlite3
import json

class SQLitePersistence:
    """
    SQLite-based persistence layer for MemMachine.
    Provides a lightweight local storage option for agent memory.
    """
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def save(self, memory_id, content, metadata=None):
        self.cursor.execute(
            "INSERT OR REPLACE INTO memory (id, content, metadata) VALUES (?, ?, ?)",
            (memory_id, content, json.dumps(metadata))
        )
        self.conn.commit()

    def load(self, memory_id):
        self.cursor.execute("SELECT content, metadata FROM memory WHERE id = ?", (memory_id,))
        row = self.cursor.fetchone()
        if row:
            return row[0], json.loads(row[1])
        return None, None
