"""
Skill 自动提取器 —— 后台 Review Agent 的核心

从对话日志中挖掘可沉淀的 Skill 模式：
1. 工具序列模式挖掘：出现 >=3 次的相同工具调用链 → 候选 Skill
2. LLM 辅助模式识别：对高频序列调用 LLM 生成人类可读描述
3. 用户反馈信号：用户表扬/纠正的操作 → 加权重/标记为反模式
"""

from typing import List, Dict, Optional
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class SkillPattern:
    """提取到的 Skill 候选模式"""
    name: str                    # 候选 Skill 名称
    description: str             # LLM 生成的描述
    tool_sequence: List[str]     # 工具调用链
    example_input: str           # 示例输入
    example_output: str          # 示例输出
    confidence: float = 0.5      # 置信度 0.0-1.0
    frequency: int = 0           # 出现次数
    sources: List[str] = field(default_factory=list)  # 来源 session_id


EXTRACTION_PROMPT = """分析以下高频出现的工具调用模式，为其生成一个 Skill 定义。

## 工具调用链
{tool_chain}

## 出现次数: {frequency}次
## 示例输入/输出:
{examples}

## 请生成:
1. **name**: 简短英文名 (kebab-case)
2. **description**: 一句话中文描述
3. **body**: 使用步骤 (1. 2. 3.) + 适用场景 + 示例

返回格式:
---
name: xxx
description: xxx
---

# Skill 标题
## 使用步骤
1. ...
"""


class SkillExtractor:
    """对话模式挖掘器"""

    def __init__(self, llm=None):
        self.llm = llm

    def extract_patterns(self, session_logs: List[Dict]) -> List[SkillPattern]:
        """
        从会话日志中提取 Skill 候选

        session_logs: [{"session_id":..., "messages":[...]}, ...]
        """
        patterns = []

        # 1. 提取所有工具调用序列
        all_sequences = self._extract_tool_sequences(session_logs)

        # 2. 找高频序列（>=3次）
        freq_sequences = self._find_frequent_sequences(all_sequences, min_support=3)

        # 3. 为每个高频序列生成 SkillPattern
        for seq_tuple, freq in freq_sequences.items():
            seq = list(seq_tuple)
            examples = self._find_examples(session_logs, seq)

            # LLM 辅助生成描述
            if self.llm and len(seq) > 1:
                description = self._llm_describe(seq, freq, examples)
            else:
                description = f"自动提取: {' → '.join(seq)} (出现{freq}次)"

            patterns.append(SkillPattern(
                name="auto-" + "-".join(seq[:3]).lower().replace("_", "-"),
                description=description,
                tool_sequence=seq,
                example_input=examples[0]["input"] if examples else "",
                example_output=examples[0]["output"] if examples else "",
                confidence=min(freq / 10, 0.9),
                frequency=freq,
                sources=[ex["session_id"] for ex in examples],
            ))

        return patterns

    def _extract_tool_sequences(self, session_logs: List[Dict]) -> List[Dict]:
        """从会话日志中提取工具调用链"""
        sequences = []

        for log in session_logs:
            messages = log.get("messages", [])
            # 提取连续的 tool_calls 名称
            seq = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        name = tc.get("function", {}).get("name", tc.get("name", ""))
                        if name:
                            seq.append(name)

            if len(seq) >= 2:
                sequences.append({
                    "session_id": log.get("session_id", ""),
                    "sequence": seq,
                    "messages": messages,
                })

        return sequences

    def _find_frequent_sequences(self, sequences: List[Dict],
                                  min_support: int = 3) -> Dict:
        """
        找高频连续子序列

        简单实现：统计所有长度为 2-5 的连续子序列
        """
        counter = Counter()

        for seq_data in sequences:
            seq = seq_data["sequence"]
            for window in range(2, min(len(seq) + 1, 6)):
                for i in range(len(seq) - window + 1):
                    subseq = tuple(seq[i:i + window])
                    counter[subseq] += 1

        return {k: v for k, v in counter.items() if v >= min_support}

    def _find_examples(self, session_logs: List[Dict],
                        target_seq: List[str]) -> List[Dict]:
        """找到包含目标序列的示例"""
        examples = []
        for log in session_logs:
            messages = log.get("messages", [])
            # 检查是否包含这个序列
            all_names = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        name = tc.get("function", {}).get("name", tc.get("name", ""))
                        all_names.append(name)

            # 简单子序列检测
            for i in range(len(all_names) - len(target_seq) + 1):
                if all_names[i:i + len(target_seq)] == target_seq:
                    # 找到第一个 user 消息作为输入
                    user_msg = next(
                        (m for m in messages if m.get("role") == "user"), {}
                    )
                    # 找到最后一个 assistant 消息作为输出
                    assistant_msgs = [
                        m for m in messages
                        if m.get("role") == "assistant" and not m.get("tool_calls")
                    ]
                    last_answer = assistant_msgs[-1] if assistant_msgs else {}

                    examples.append({
                        "session_id": log.get("session_id", ""),
                        "input": user_msg.get("content", "")[:200],
                        "output": last_answer.get("content", "")[:200],
                    })
                    break

            if len(examples) >= 3:
                break

        return examples

    def _llm_describe(self, seq: List[str], freq: int, examples: List[Dict]) -> str:
        """调用 LLM 生成 Skill 描述"""
        try:
            examples_text = "\n".join([
                f"输入: {ex.get('input', '')}\n输出: {ex.get('output', '')}"
                for ex in examples[:3]
            ])

            response = self.llm.invoke(
                messages=[{
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(
                        tool_chain=" → ".join(seq),
                        frequency=freq,
                        examples=examples_text[:1000],
                    )
                }],
                max_tokens=400,
                temperature=0.3,
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception:
            return f"自动提取: {' → '.join(seq)} (出现{freq}次)"
