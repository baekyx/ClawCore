"""
Layer 1: Budget 截断

当上下文 Token 数超过窗口 80% 阈值时触发，保留最近 N 轮完整对话 + 系统消息。
核心策略和 HelloAgents HistoryManager.compress() 一样，但增加了 Token 感知能力。
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple

HELLO_AGENTS_PATH = Path("D:/code/HelloAgents")
if str(HELLO_AGENTS_PATH) not in sys.path:
    sys.path.insert(0, str(HELLO_AGENTS_PATH))

from config.settings import AgentConfig


class BudgetTruncator:
    """Layer 1: Token 预算感知的截断器"""

    def __init__(self, config: AgentConfig):
        self.context_window = config.context_window
        self.threshold = config.compression_threshold
        self.min_retain_rounds = config.min_retain_rounds

    def needs_truncation(self, messages: List[Dict]) -> bool:
        """检查是否超过预算阈值"""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4  # 粗略估算 1token≈4字符
        return estimated_tokens > self.context_window * self.threshold

    def truncate(self, messages: List[Dict]) -> List[Dict]:
        """
        截断策略：
        1. 保留 system 消息（索引0）
        2. 保留最近 min_retain_rounds 轮对话
        3. 中间插入一条 summary 消息作为标记
        """
        if not self.needs_truncation(messages):
            return messages

        # 分离 system 消息
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]

        # 找到所有 user 消息的索引（轮次边界）
        user_indices = [i for i, m in enumerate(other_msgs) if m["role"] == "user"]

        if len(user_indices) <= self.min_retain_rounds:
            return messages  # 还不够多，不截断

        # 保留最近 N 轮
        keep_from = user_indices[-self.min_retain_rounds]

        kept = other_msgs[keep_from:]
        truncated_count = len(other_msgs) - len(kept)

        result = system_msgs + [{
            "role": "system",
            "content": f"[上下文压缩] 之前的 {truncated_count} 条消息已截断，以下是最近的对话"
        }] + kept

        return result
