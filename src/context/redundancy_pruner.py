"""
Layer 2: 冗余裁剪

规则引擎检测并移除：
1. 重复的工具输出（相同工具返回完全相同的结果）
2. 连续的相似消息（内容重叠 > 80%）
3. 已解决的问题（用户明确确认，之前的往返可以压缩）

纯规则实现，无需 LLM 调用，O(n) 时间复杂度
"""

import hashlib
from typing import List, Dict, Set


class RedundancyPruner:
    """Layer 2: 冗余检测与裁剪"""

    def __init__(self, duplicate_threshold: float = 0.85):
        self.duplicate_threshold = duplicate_threshold

    def prune(self, messages: List[Dict]) -> List[Dict]:
        """
        三步裁剪：
        1. 去重：相同 hash 的 tool 输出只保留第一次
        2. 合并：连续的 tool 消息如果内容高度相似则合并
        3. 修剪已解决：用户说"谢谢"/"好的"后，前面的调试性往返可以简化
        """
        messages = self._deduplicate_tool_outputs(messages)
        messages = self._merge_similar_tool_outputs(messages)
        messages = self._trim_resolved_threads(messages)
        return messages

    def _deduplicate_tool_outputs(self, messages: List[Dict]) -> List[Dict]:
        """去重：相同 tool 输出只保留第一次出现"""
        seen_hashes: Set[str] = set()
        result = []

        for msg in messages:
            if msg["role"] == "tool":
                content_hash = hashlib.md5(
                    msg.get("content", "").encode()
                ).hexdigest()

                if content_hash in seen_hashes:
                    continue  # 跳过重复的工具输出

                seen_hashes.add(content_hash)

            result.append(msg)

        return result

    def _merge_similar_tool_outputs(self, messages: List[Dict]) -> List[Dict]:
        """合并连续相似的 tool 输出"""
        result = []
        i = 0

        while i < len(messages):
            msg = messages[i]

            # 检查是否是一串连续的 tool 输出
            if msg["role"] == "tool":
                group = [msg]
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "tool":
                    similarity = self._text_similarity(
                        msg.get("content", ""),
                        messages[j].get("content", "")
                    )
                    if similarity > self.duplicate_threshold:
                        group.append(messages[j])
                        j += 1
                    else:
                        break

                if len(group) > 1:
                    # 合并：只保留第一条 + 一条摘要
                    result.append(group[0])
                    result.append({
                        "role": "tool",
                        "content": f"[合并] 后续 {len(group)-1} 条相似的工具输出已省略",
                        "tool_call_id": group[-1].get("tool_call_id", "")
                    })
                    i = j
                else:
                    result.append(msg)
                    i += 1
            else:
                result.append(msg)
                i += 1

        return result

    def _trim_resolved_threads(self, messages: List[Dict]) -> List[Dict]:
        """修剪已解决的对话：用户确认后，前面的重试/纠错往返可压缩"""
        confirmation_keywords = ["谢谢", "好的", "可以了", "没问题", "明白了", "got it", "thanks"]
        result = []

        for i, msg in enumerate(messages):
            if msg["role"] == "user" and any(
                kw in msg.get("content", "") for kw in confirmation_keywords
            ):
                # 找到这个确认之前的最近一个 user 消息
                prev_user_idx = None
                for j in range(i - 1, -1, -1):
                    if messages[j]["role"] == "user":
                        prev_user_idx = j
                        break

                if prev_user_idx is not None and (i - prev_user_idx) > 3:
                    # 中间有较长的往返，插入压缩标记
                    result.append({
                        "role": "system",
                        "content": "[上下文压缩] 中间的纠错/调试往返已省略"
                    })

            result.append(msg)

        return result

    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单 Jaccard 相似度（基于字符 4-gram）"""
        if not text1 or not text2:
            return 0.0

        def ngrams(s, n=4):
            return set(s[i:i+n] for i in range(len(s) - n + 1))

        set1 = ngrams(text1[:200])
        set2 = ngrams(text2[:200])

        if not set1 or not set2:
            return 0.0

        return len(set1 & set2) / len(set1 | set2)
