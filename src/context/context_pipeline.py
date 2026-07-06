"""
上下文压缩流水线 — 四层编排引擎

四层由快到慢、由轻到重递进：
Layer 1: Budget 截断 (O(n), 免费, 低损失)          — 超预算时保留最近N轮
Layer 2: 冗余裁剪 (O(n), 免费, 极低损失)             — 去重+合并相似输出
Layer 3: 结构化精缩 (O(n)+LLM, 低成本, 中损失)      — LLM生成结构化摘要
Layer 4: 自动阈值 (O(1), 免费, 自适应)               — 动态调整压缩参数
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

HELLO_AGENTS_PATH = Path("D:/code/HelloAgents")
if str(HELLO_AGENTS_PATH) not in sys.path:
    sys.path.insert(0, str(HELLO_AGENTS_PATH))

from config.settings import AgentConfig
from .budget_truncator import BudgetTruncator
from .redundancy_pruner import RedundancyPruner
from .structural_compressor import StructuralCompressor
from .auto_threshold import AutoThreshold, CompressionParams


class ContextPipeline:
    """
    四层上下文压缩流水线

    用法：
        pipeline = ContextPipeline(config, llm)
        if pipeline.should_compress(messages):
            messages = pipeline.compress(messages)
    """

    def __init__(self, config: AgentConfig, llm=None):
        self.config = config

        # 四层组件
        self.budget = BudgetTruncator(config)
        self.redundancy = RedundancyPruner()
        self.structural = StructuralCompressor(config, llm)
        self.auto = AutoThreshold()

        # 统计
        self.stats = {
            "l1_truncations": 0,
            "l2_pruned": 0,
            "l3_compressions": 0,
            "total_rounds": 0,
        }

    def should_compress(self, messages: List[Dict]) -> bool:
        """是否触发压缩"""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4
        params = self.auto.adjust(messages)
        return estimated_tokens > self.config.context_window * params.threshold

    def compress(self, messages: List[Dict]) -> List[Dict]:
        """执行四层压缩流水线，每层异常时降级到下一层"""
        try:
            params = self.auto.adjust(messages)
        except Exception:
            params = None

        original_count = len(messages)

        # Layer 1: Budget 截断
        try:
            if self.budget.needs_truncation(messages):
                messages = self.budget.truncate(messages)
                self.stats["l1_truncations"] += 1
        except Exception as e:
            print(f"[ContextPipeline] L1 截断失败,降级跳过: {e}")

        # Layer 2: 冗余裁剪
        try:
            pre_prune = len(messages)
            messages = self.redundancy.prune(messages)
            if len(messages) < pre_prune:
                self.stats["l2_pruned"] += 1
        except Exception as e:
            print(f"[ContextPipeline] L2 裁剪失败,降级跳过: {e}")

        # Layer 3: 结构化精缩
        try:
            total_chars = sum(len(m.get("content", "")) for m in messages)
            estimated_tokens = total_chars // 4
            kr = params.keep_recent_rounds if params else 5
            if self.structural.needs_compression(messages, estimated_tokens):
                messages = self.structural.compress(messages, keep_recent_rounds=kr)
                self.stats["l3_compressions"] += 1
        except Exception as e:
            print(f"[ContextPipeline] L3 摘要失败,降级跳过: {e}")

        self.stats["total_rounds"] += 1

        # 日志
        reduction = (
            (1 - len(messages) / original_count) * 100 if original_count > 0 else 0
        )
        if reduction > 5:
            print(
                f"[ContextPipeline] 压缩: {original_count}→{len(messages)} "
                f"({reduction:.0f}%) | L1={self.stats['l1_truncations']} "
                f"L2={self.stats['l2_pruned']} L3={self.stats['l3_compressions']}"
            )

        return messages

    def get_compression_ratio(self) -> float:
        """获取累计压缩率（仅统计 L2/L3 的减少量）"""
        return (
            self.stats["total_rounds"]
            and (self.stats["l2_pruned"] + self.stats["l3_compressions"])
            / max(self.stats["total_rounds"], 1)
        )
