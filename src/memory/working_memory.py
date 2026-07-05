"""
Layer 2: 工作记忆 (Postgres JSONB)

存储跨会话的"当前关注事项"：
- User 命名空间：用户偏好、个人信息、常用设置
- Task  命名空间：未完成任务、当前项目上下文
- Memory 命名空间：Agent 认为应该记住的重要信息

每条记录带 importance 分数，高分的会被沉淀到 Layer 3 长期记忆
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

try:
    import psycopg2
    import psycopg2.extras
    HAS_PG = True
except ImportError:
    psycopg2 = None
    HAS_PG = False

from config.settings import PostgresConfig


@dataclass
class WorkingMemoryItem:
    namespace: str     # User | Task | Memory
    key: str           # 唯一标识（如 "preferred_language"）
    title: str         # 简短标题
    content: str       # 完整内容
    importance: float = 0.5  # 0.0-1.0
    metadata: Dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class WorkingMemory:
    """Layer 2: Postgres JSONB 工作记忆"""

    def __init__(self, config: PostgresConfig):
        self.config = config
        self._init_db()

    def _get_conn(self):
        if not HAS_PG:
            raise RuntimeError("psycopg2 未安装")
        return psycopg2.connect(
            host=self.config.host, port=self.config.port,
            database=self.config.database,
            user=self.config.user, password=self.config.password
        )

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS working_memory (
                    id SERIAL PRIMARY KEY,
                    namespace VARCHAR(32) NOT NULL DEFAULT 'Memory',
                    key VARCHAR(128) NOT NULL,
                    title VARCHAR(256) NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    importance FLOAT NOT NULL DEFAULT 0.5,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(namespace, key)
                );
                CREATE INDEX IF NOT EXISTS idx_wm_namespace ON working_memory(namespace);
                CREATE INDEX IF NOT EXISTS idx_wm_importance ON working_memory(importance DESC);
            """)

    def upsert(self, item: WorkingMemoryItem) -> int:
        """插入或更新一条工作记忆，返回 id"""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            row = conn.execute("""
                INSERT INTO working_memory (namespace, key, title, content, importance, metadata, created_at, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (namespace, key) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    importance = EXCLUDED.importance,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                RETURNING id
            """, (item.namespace, item.key, item.title, item.content,
                  item.importance, json_dumps(item.metadata), now, now))
            return row.fetchone()[0]

    def get(self, namespace: str, key: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM working_memory WHERE namespace=%s AND key=%s",
                (namespace, key)
            ).fetchone()
        return dict(row) if row else None

    def search(self, namespace: str = None, keyword: str = "") -> List[Dict]:
        """按命名空间 + 关键词搜索"""
        with self._get_conn() as conn:
            if namespace and keyword:
                rows = conn.execute(
                    """SELECT * FROM working_memory
                       WHERE namespace=%s AND (title ILIKE %s OR content ILIKE %s)
                       ORDER BY importance DESC LIMIT 20""",
                    (namespace, f"%{keyword}%", f"%{keyword}%")
                ).fetchall()
            elif namespace:
                rows = conn.execute(
                    "SELECT * FROM working_memory WHERE namespace=%s ORDER BY importance DESC LIMIT 20",
                    (namespace,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM working_memory ORDER BY importance DESC LIMIT 20"
                ).fetchall()
        return [dict(r) for r in rows]

    def get_high_importance(self, threshold: float = 0.7) -> List[Dict]:
        """获取高分记忆（用于沉淀到 Layer 3）"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM working_memory WHERE importance >= %s ORDER BY importance DESC",
                (threshold,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_user_profile(self) -> str:
        """获取用户画像文本（用于注入 System Prompt）"""
        items = self.search(namespace="User")
        if not items:
            return ""
        lines = ["## 用户信息"]
        for item in items:
            lines.append(f"- {item['title']}: {item['content'][:200]}")
        return "\n".join(lines)

    def get_pending_tasks(self) -> str:
        """获取待办任务文本"""
        items = self.search(namespace="Task")
        if not items:
            return ""
        lines = ["## 进行中的任务"]
        for item in items:
            lines.append(f"- {item['title']}")
        return "\n".join(lines)


def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
