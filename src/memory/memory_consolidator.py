"""
记忆沉淀器 — 后台异步运行

任务：
1. 会话结束后，LLM 提取关键信息 → Layer 2 工作记忆
2. 定期将高分工作记忆向量化 → Layer 3 长期记忆
"""

import asyncio
from typing import List, Dict, Optional
from datetime import datetime

from .memory_manager import MemoryManager
from .working_memory import WorkingMemoryItem


class MemoryConsolidator:
    """记忆沉淀器 — 后台运行"""

    def __init__(self, memory_manager: MemoryManager, llm=None):
        self.memory = memory_manager
        self.llm = llm  # Phase 2: LLM 辅助提取

    def consolidate_session(self, session_id: str, messages: List[Dict]) -> int:
        """
        会话结束后的沉淀：从完整对话中提取关键信息

        策略（Phase 2 使用规则 + LLM辅助）：
        1. 检测用户明确说"记住"、"别忘了"等关键词 → importance=0.9
        2. 检测决策类对话 → importance=0.7
        3. 调用 LLM 提取关键实体和偏好
        """
        if not self.memory.working:
            return 0

        count = 0

        # 规则提取：匹配"记住xxx"模式
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            if role == "user" and any(kw in content for kw in ["记住", "别忘了", "我是", "我喜欢", "我不喜欢"]):
                self.memory.remember_fact(
                    title=f"用户偏好: {content[:50]}",
                    content=content,
                    importance=0.85
                )
                count += 1

            if role == "assistant" and any(kw in content for kw in ["总结", "结论", "决定", "方案"]):
                self.memory.remember_fact(
                    title=f"决策记录: {content[:50]}",
                    content=content[:500],
                    importance=0.7
                )
                count += 1

        return count

    async def run_periodic_consolidation(self):
        """定期沉淀：工作记忆 → 长期记忆"""
        while True:
            try:
                result = self.memory.consolidate()
                if result["working_to_longterm"] > 0:
                    print(f"[MemoryConsolidator] 沉淀 {result['working_to_longterm']} 条到长期记忆")
            except Exception as e:
                print(f"[MemoryConsolidator] 沉淀失败: {e}")
            await asyncio.sleep(3600)  # 每小时一次

    def auto_consolidate_session(self, answers: List[str]) -> int:
        """自动沉淀：从 Agent 的答案中提取可记忆的事实"""
        count = 0
        for answer in answers:
            if len(answer) > 20:
                self.memory.remember_fact(
                    title=f"历史回答: {answer[:50]}",
                    content=answer[:300],
                    importance=0.5
                )
                count += 1
        return count
