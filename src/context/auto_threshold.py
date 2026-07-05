"""
Layer 4: 自动阈值裁剪

根据任务复杂度动态调整压缩参数，避免过度或不足压缩。

启发式规则：
- 简单查询（单轮可完成）→ 高阈值（80%），延迟压缩
- 多步推理（需要3+轮）→ 中阈值（70%），适度压缩
- 长文档分析（输出很大）→ 低阈值（60%），积极压缩
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class CompressionParams:
    threshold: float        # 压缩触发阈值 (0.0-1.0)
    min_retain_rounds: int  # 最少保留轮次
    keep_recent_rounds: int # Layer 3 保留轮次


class AutoThreshold:
    """Layer 4: 自适应阈值调整"""

    DEFAULT_PARAMS = CompressionParams(
        threshold=0.80, min_retain_rounds=10, keep_recent_rounds=5
    )

    def adjust(self, messages: List[Dict]) -> CompressionParams:
        """
        根据消息特征动态调整压缩参数

        检测维度：
        1. 对话轮次 → 影响阈值
        2. 工具调用次数 → 影响保留轮次
        3. 平均消息长度 → 影响压缩激进程度
        """
        rounds = self._count_rounds(messages)
        tool_calls = self._count_tool_calls(messages)
        avg_length = self._avg_message_length(messages)

        # 默认参数
        params = CompressionParams(
            threshold=0.80,
            min_retain_rounds=10,
            keep_recent_rounds=5,
        )

        # 多轮对话 → 降低阈值（更早触发压缩）
        if rounds > 20:
            params.threshold = 0.60
        elif rounds > 10:
            params.threshold = 0.70

        # 多工具调用 → 减少保留轮次（每轮信息量大）
        if tool_calls > 15:
            params.min_retain_rounds = 5
            params.keep_recent_rounds = 3
        elif tool_calls > 8:
            params.min_retain_rounds = 7
            params.keep_recent_rounds = 4

        # 长消息（如文档内容）→ 更激进压缩
        if avg_length > 2000:
            params.threshold -= 0.05
            params.min_retain_rounds = min(params.min_retain_rounds, 5)

        return params

    def _count_rounds(self, messages: List[Dict]) -> int:
        return sum(1 for m in messages if m["role"] == "user")

    def _count_tool_calls(self, messages: List[Dict]) -> int:
        return sum(1 for m in messages if m["role"] == "tool")

    def _avg_message_length(self, messages: List[Dict]) -> float:
        if not messages:
            return 0
        return sum(len(m.get("content", "")) for m in messages) / len(messages)
