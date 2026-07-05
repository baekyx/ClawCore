"""
记忆管理器 — 统一管理三层记忆

对外暴露简洁 API：
- retrieve(query) → 检索相关记忆文本
- remember(key_info) → 跨层写入记忆
- consolidate() → 触发沉淀任务
"""

from typing import List, Dict, Optional

from config.settings import MyClawConfig, get_config
from .session_memory import SessionMemory, SessionMessage
from .working_memory import WorkingMemory, WorkingMemoryItem
from .long_term_memory import LongTermMemory


class MemoryManager:
    """统一记忆管理入口"""

    def __init__(self, config: MyClawConfig = None):
        self.config = config or get_config()

        # Layer 1: 会话态
        self.session = SessionMemory(db_path="data/sessions.db")

        # Layer 2: 工作记忆 (Postgres)
        try:
            self.working = WorkingMemory(self.config.postgres)
        except Exception as e:
            print(f"[Memory]  工作记忆(Postgres)不可用: {e}")
            self.working = None

        # Layer 3: 长期记忆 (Postgres pgvector)
        try:
            self.long_term = LongTermMemory(
                self.config.postgres, self.config.embedding, self.config.retrieval
            )
        except Exception as e:
            print(f"[Memory]  长期记忆(pgvector)不可用: {e}")
            self.long_term = None

    # === 检索 ===

    def retrieve(self, query: str) -> str:
        """检索与 query 相关的记忆，返回可注入 LLM 的文本"""
        parts = []

        # Layer 2: 工作记忆中的用户画像 + 任务
        if self.working:
            profile = self.working.get_user_profile()
            if profile:
                parts.append(profile)
            tasks = self.working.get_pending_tasks()
            if tasks:
                parts.append(tasks)

        # Layer 3: 长期记忆混合检索
        if self.long_term and self.long_term.count() > 0:
            ltm_context = self.long_term.get_context_for_query(query)
            if ltm_context:
                parts.append(ltm_context)

        return "\n\n".join(parts) if parts else ""

    # === 写入 ===

    def remember_session_message(self, role: str, content: str):
        """记录会话消息到 Layer 1"""
        self.session.add_message(SessionMessage(role=role, content=content))

    def remember_fact(self, title: str, content: str, importance: float = 0.5):
        """记住一个事实到工作记忆"""
        if self.working:
            key = title.lower().replace(" ", "_")[:128]
            self.working.upsert(WorkingMemoryItem(
                namespace="Memory", key=key, title=title,
                content=content, importance=importance,
            ))

    def remember_user_preference(self, key: str, value: str):
        """记住用户偏好"""
        if self.working:
            self.working.upsert(WorkingMemoryItem(
                namespace="User", key=key, title=key.replace("_", " ").title(),
                content=value, importance=0.8,
            ))

    # === 沉淀 ===

    def consolidate(self):
        """触发记忆沉淀 (Session → Working → Long-term)

        规则：
        1. 高频出现的信息 importance +0.1
        2. importance >= 0.7 → 沉淀到长期记忆
        """
        results = {"working_to_longterm": 0}

        if self.working and self.long_term:
            high_imp = self.working.get_high_importance(threshold=0.7)
            if high_imp:
                items = [{
                    "content": item["content"],
                    "memory_type": item["namespace"].lower(),
                    "importance": item["importance"],
                    "metadata": {"source": "working_memory", "wm_key": item["key"]},
                } for item in high_imp]
                ids = self.long_term.add_batch(items)
                results["working_to_longterm"] = len(ids)

        return results
