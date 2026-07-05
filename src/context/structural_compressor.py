"""
Layer 3: 结构化精缩

当 Layer 1+2 不够时（仍然超预算），调用轻量 LLM 生成结构化摘要。
复用 HelloAgents Agent._generate_smart_summary() 的 Prompt 模式。

输出格式：
- 任务目标
- 关键决策
- 已完成工作
- 待处理事项
- 重要发现
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

HELLO_AGENTS_PATH = Path("D:/code/HelloAgents")
if str(HELLO_AGENTS_PATH) not in sys.path:
    sys.path.insert(0, str(HELLO_AGENTS_PATH))

from config.settings import AgentConfig


COMPRESSION_PROMPT = """请将以下对话历史压缩为结构化摘要，保留关键信息：

## 对话历史
{history_text}

## 摘要要求
1. **任务目标**：用户想要完成什么？
2. **关键决策**：做了哪些重要决定？
3. **已完成工作**：完成了哪些任务？（列表形式，最多5条）
4. **待处理事项**：还有什么未完成？
5. **重要发现**：有哪些关键信息或问题？

请用简洁的中文输出，每部分不超过3行。"""


class StructuralCompressor:
    """Layer 3: LLM 驱动的结构化摘要"""

    def __init__(self, config: AgentConfig, llm=None):
        self.config = config
        self.llm = llm  # 轻量 LLM（如 deepseek-chat）

    def needs_compression(self, messages: List[Dict], token_count: int) -> bool:
        """Layer 1+2 之后仍超过阈值才触发 Layer 3"""
        return token_count > self.config.context_window * self.config.compression_threshold

    def compress(self, messages: List[Dict], keep_recent_rounds: int = 5) -> List[Dict]:
        """
        结构化压缩：
        1. 分离"旧消息"和"最近消息"
        2. 对旧消息调用 LLM 生成结构化摘要
        3. 返回：[system] + [摘要] + [最近消息]
        """
        if not self.llm or len(messages) < 10:
            return self._simple_truncate(messages, keep_recent_rounds)

        # 找到保留起点（最近 N 轮）
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if len(user_indices) <= keep_recent_rounds:
            return messages

        keep_from = user_indices[-keep_recent_rounds]
        old_messages = messages[:keep_from]
        recent_messages = messages[keep_from:]

        # 格式化旧消息
        history_text = "\n".join([
            f"[{m['role']}] {m.get('content', '')[:300]}"
            for m in old_messages
            if m["role"] not in ("system",)
        ])

        # 调用 LLM 生成摘要
        try:
            summary = self.llm.invoke(
                messages=[{
                    "role": "system",
                    "content": "你是一个专业的对话摘要助手，擅长提取关键信息。"
                }, {
                    "role": "user",
                    "content": COMPRESSION_PROMPT.format(history_text=history_text)
                }],
                max_tokens=500,
                temperature=0.3,
            )
            summary_text = summary.content if hasattr(summary, 'content') else str(summary)
        except Exception:
            # LLM 失败时降级为简单截断
            return self._simple_truncate(messages, keep_recent_rounds)

        # 构建结果
        system_msgs = [m for m in messages if m["role"] == "system" and "压缩" not in m.get("content", "")]

        result = system_msgs + [{
            "role": "system",
            "content": f"## 历史摘要\n{summary_text}"
        }] + recent_messages

        return result

    def _simple_truncate(self, messages: List[Dict], keep_recent_rounds: int = 5) -> List[Dict]:
        """降级方案：简单截断（不用 LLM）"""
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if len(user_indices) <= keep_recent_rounds:
            return messages

        keep_from = user_indices[-keep_recent_rounds]
        system_msgs = [m for m in messages if m["role"] == "system"]
        recent = messages[keep_from:]

        return system_msgs + [{
            "role": "system",
            "content": f"之前的 {keep_from} 条消息已省略"
        }] + recent
