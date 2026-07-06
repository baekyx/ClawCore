"""
Layer 2: 工作记忆 (Postgres JSONB)
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

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
    namespace: str
    key: str
    title: str
    content: str
    importance: float = 0.5
    metadata: Dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class WorkingMemory:
    """Layer 2: Postgres JSONB 工作记忆"""

    def __init__(self, config: PostgresConfig):
        self.config = config
        self._conn = None
        self._init_db()

    @contextmanager
    def _cursor(self):
        if not HAS_PG:
            raise RuntimeError("psycopg2 未安装")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._conn is None or self._conn.closed:
                    self._conn = psycopg2.connect(
                        host=self.config.host, port=self.config.port,
                        database=self.config.database,
                        user=self.config.user, password=self.config.password
                    )
                cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                try:
                    yield cur
                    self._conn.commit()
                    return
                finally:
                    cur.close()
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                if self._conn:
                    self._conn.close()
                self._conn = None
                if attempt == max_retries - 1:
                    raise RuntimeError(f"PG 重连失败(已重试{max_retries}次): {e}") from e
                import time
                time.sleep(1.0 * (attempt + 1))

    def _init_db(self):
        with self._cursor() as cur:
            cur.execute("""
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
        now = datetime.now().isoformat()
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO working_memory (namespace, key, title, content, importance, metadata, created_at, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (namespace, key) DO UPDATE SET
                    title = EXCLUDED.title, content = EXCLUDED.content,
                    importance = EXCLUDED.importance, metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                RETURNING id
            """, (item.namespace, item.key, item.title, item.content,
                  item.importance, _json(item.metadata), now, now))
            return cur.fetchone()["id"]

    def get(self, namespace: str, key: str) -> Optional[Dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM working_memory WHERE namespace=%s AND key=%s",
                (namespace, key))
            row = cur.fetchone()
        return dict(row) if row else None

    def search(self, namespace: str = None, keyword: str = "") -> List[Dict]:
        with self._cursor() as cur:
            if namespace and keyword:
                cur.execute(
                    """SELECT * FROM working_memory
                       WHERE namespace=%s AND (title ILIKE %s OR content ILIKE %s)
                       ORDER BY importance DESC LIMIT 20""",
                    (namespace, f"%{keyword}%", f"%{keyword}%"))
            elif namespace:
                cur.execute(
                    "SELECT * FROM working_memory WHERE namespace=%s ORDER BY importance DESC LIMIT 20",
                    (namespace,))
            else:
                cur.execute(
                    "SELECT * FROM working_memory ORDER BY importance DESC LIMIT 20")
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_high_importance(self, threshold: float = 0.7) -> List[Dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM working_memory WHERE importance >= %s ORDER BY importance DESC",
                (threshold,))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_user_profile(self) -> str:
        items = self.search(namespace="User")
        if not items:
            return ""
        return "\n".join([f"- {item['title']}: {item['content'][:200]}" for item in items])


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
