"""测试上下文压缩 L1+L2（纯规则，无需 LLM）"""
import pytest
from config.settings import AgentConfig
from src.context.budget_truncator import BudgetTruncator
from src.context.redundancy_pruner import RedundancyPruner
from src.context.auto_threshold import AutoThreshold


# --- BudgetTruncator ---

class TestBudgetTruncator:
    def test_no_truncation_needed_for_short_messages(self):
        config = AgentConfig(context_window=128000, compression_threshold=0.8)
        trunc = BudgetTruncator(config)
        msgs = [{"role": "user", "content": "短消息"}]
        assert not trunc.needs_truncation(msgs)

    def test_truncation_needed_for_long_messages(self):
        config = AgentConfig(context_window=1000, compression_threshold=0.8)
        trunc = BudgetTruncator(config)
        msgs = [{"role": "user", "content": "x" * 5000}]
        assert trunc.needs_truncation(msgs)

    def test_truncate_preserves_system_message(self):
        config = AgentConfig(context_window=2000, compression_threshold=0.5, min_retain_rounds=2)
        trunc = BudgetTruncator(config)
        msgs = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "x" * 300},
            {"role": "assistant", "content": "y" * 300},
            {"role": "user", "content": "x" * 300},
            {"role": "assistant", "content": "y" * 300},
            {"role": "user", "content": "最新问题"},
            {"role": "assistant", "content": "最新回答"},
        ]
        result = trunc.truncate(msgs)
        # system 消息应保留
        assert any(m["role"] == "system" for m in result)
        # 最近 2 轮应保留
        user_msgs = [m for m in result if m["role"] == "user"]
        assert "最新问题" in user_msgs[-1]["content"]


# --- RedundancyPruner ---

class TestRedundancyPruner:
    def setup_method(self):
        self.pruner = RedundancyPruner()

    def test_deduplicate_identical_tool_outputs(self):
        msgs = [
            {"role": "tool", "content": "结果A", "tool_call_id": "1"},
            {"role": "tool", "content": "结果A", "tool_call_id": "2"},  # 重复
            {"role": "tool", "content": "结果B", "tool_call_id": "3"},
        ]
        result = self.pruner.prune(msgs)
        assert len(result) == 2  # 重复的应被移除

    def test_no_dedup_for_different_outputs(self):
        msgs = [
            {"role": "tool", "content": "结果A", "tool_call_id": "1"},
            {"role": "tool", "content": "结果B", "tool_call_id": "2"},
        ]
        result = self.pruner.prune(msgs)
        assert len(result) == 2

    def test_merge_similar_tool_outputs(self):
        # 两条几乎一模一样的内容（只有结尾不同）
        base = "第1行: import os\\n第2行: import sys\\n第3行: x=1\\n第4行: y=2\\n第5行: same\\n第6行: same\\n第7行: same\\n第8行: same\\n第9行: same\\n第10行: same"
        msgs = [
            {"role": "user", "content": "查文件"},
            {"role": "tool", "content": base + "\\n第11行: a"},
            {"role": "tool", "content": base + "\\n第11行: b"},
            {"role": "tool", "content": base + "\\n第11行: c"},
        ]
        result = self.pruner.prune(msgs)
        tool_count = sum(1 for m in result if m["role"] == "tool")
        assert tool_count < 3  # 3条相似输出→合并后少于3条

    def test_trim_confirmed_threads(self):
        # 中间有 4 条往返消息，超过阈值(>3)触发压缩标记
        msgs = [
            {"role": "user", "content": "帮我做X"},
            {"role": "assistant", "content": "尝试方法1"},
            {"role": "assistant", "content": "不对修正"},
            {"role": "assistant", "content": "再次尝试"},
            {"role": "assistant", "content": "纠正数据"},
            {"role": "user", "content": "谢谢，好了"},
        ]
        result = self.pruner.prune(msgs)
        assert any(m["role"] == "system" for m in result)


# --- AutoThreshold ---

class TestAutoThreshold:
    def setup_method(self):
        self.auto = AutoThreshold()

    def test_default_params_for_short_conversation(self):
        msgs = [{"role": "user", "content": "你好"}] * 2
        p = self.auto.adjust(msgs)
        assert p.threshold == 0.80
        assert p.min_retain_rounds == 10

    def test_lower_threshold_for_long_conversation(self):
        msgs = [{"role": "user", "content": "x"}] * 25  # >20 轮
        p = self.auto.adjust(msgs)
        assert p.threshold <= 0.70  # 降低阈值

    def test_fewer_retain_rounds_for_many_tool_calls(self):
        msgs = (
            [{"role": "user", "content": "x"}]
            + [{"role": "tool", "content": "y"}] * 20
        )
        p = self.auto.adjust(msgs)
        assert p.min_retain_rounds <= 7
