"""
Layer 1: 会话态记忆 (SQLite)

存储当前会话的完整对话历史，支持：
- 会话创建/切换/删除
- 消息追加与分页读取
- 上下文窗口大小内的最近消息获取
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class SessionMessage:
    role: str  # user | assistant | tool | system
    content: str
    timestamp: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class SessionMemory:
    """Layer 1: SQLite 会话态记忆"""

    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = db_path
        self._current_session_id: Optional[str] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                CREATE INDEX IF NOT EXISTS idx_msg_session
                    ON messages(session_id, id);
            """)

    # === 会话管理 ===

    def create_session(self, title: str = "") -> str:
        """创建新会话，返回 session_id"""
        sid = f"s-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{id(self):04x}"
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at) VALUES(?,?,?,?)",
                (sid, title, now, now)
            )
        self._current_session_id = sid
        return sid

    def list_sessions(self) -> List[Dict]:
        """列出所有会话"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        if self._current_session_id == session_id:
            self._current_session_id = None

    # === 消息管理 ===

    def add_message(self, msg: SessionMessage, session_id: str = None):
        sid = session_id or self._current_session_id
        if not sid:
            sid = self.create_session()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, metadata, created_at) VALUES(?,?,?,?,?)",
                (sid, msg.role, msg.content, json.dumps(msg.metadata, ensure_ascii=False), msg.timestamp)
            )
            conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (msg.timestamp, sid))

    def get_messages(self, session_id: str = None, limit: int = 100) -> List[Dict]:
        """获取最近 N 条消息"""
        sid = session_id or self._current_session_id
        if not sid:
            return []
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata, created_at FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (sid, limit)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_recent_for_context(self, session_id: str = None, max_tokens: int = 8000) -> List[Dict]:
        """获取适合注入 LLM 上下文的最近消息（粗略估算 1token≈4字符）"""
        messages = self.get_messages(session_id)
        total_chars = 0
        recent = []
        for msg in reversed(messages):
            total_chars += len(msg["content"])
            if total_chars > max_tokens * 4:
                break
            recent.insert(0, msg)
        return recent

    @property
    def current_session_id(self) -> Optional[str]:
        return self._current_session_id
