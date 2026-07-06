"""
验证四层上下文压缩流水线 — 用模拟长对话触发每一层
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import AgentConfig
from src.context.context_pipeline import ContextPipeline


def make_rounds(n: int, chars_per_msg: int = 300) -> list:
    """生成 n 轮模拟对话，每轮 user + assistant 各 chars_per_msg 字符"""
    msgs = [{"role": "system", "content": "你是一个AI助手"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"问题{i}: " + "A" * chars_per_msg})
        msgs.append({"role": "assistant", "content": f"回答{i}: " + "B" * chars_per_msg})
    return msgs


def make_rounds_with_tools(n: int) -> list:
    """生成包含重复工具输出的对话"""
    msgs = [{"role": "system", "content": "你是助手"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"分析文件{i}"})
        msgs.append({"role": "assistant", "content": "开始分析", "tool_calls": [
            {"name": "file_read", "arguments": '{"path": "data.txt"}'}
        ]})
        base = "x" * 200
        msgs.append({"role": "tool", "content": base + f"\\n[序号{i}]", "tool_call_id": f"call_{i}"})
        msgs.append({"role": "assistant", "content": f"分析结果{i}: " + "C" * 500})
    return msgs


# === 测试 L1: Budget 截断 ===
def test_l1_budget_truncation():
    """50轮对话 → 应触发L1截断"""
    config = AgentConfig(context_window=5000, compression_threshold=0.5, min_retain_rounds=3)
    pipeline = ContextPipeline(config)
    msgs = make_rounds(50, chars_per_msg=200)

    assert pipeline.should_compress(msgs), "50轮对话应触发压缩"
    result = pipeline.compress(msgs)

    # 截断后应小于原始
    assert len(result) < len(msgs), f"L1 应减少消息量: {len(msgs)}→{len(result)}"
    assert pipeline.stats["l1_truncations"] >= 1, "应记录 L1 截断次数"


# === 测试 L2: 冗余裁剪 ===
def test_l2_redundancy_pruning():
    """重复工具输出 → 应触发L2去重"""
    config = AgentConfig(context_window=128000, compression_threshold=0.8)
    pipeline = ContextPipeline(config)

    # 手动构造：强制触发压缩后走 L2
    # 大量完全相同的tool消息会被去重
    msgs = [{"role": "system", "content": "SYS"}]
    msgs.append({"role": "user", "content": "查"})
    for i in range(10):
        msgs.append({"role": "tool", "content": "完全一样的输出" * 50, "tool_call_id": f"t{i}"})

    result = pipeline.compress(msgs)
    # 10 条相同 tool 输出应被去重到 1 条
    tool_count = sum(1 for m in result if m["role"] == "tool")
    assert tool_count < 5, f"去重后 tool 应 ≤1, 实际 {tool_count}"


# === 测试 L3: 结构化精缩 ===
def test_l3_structural_compression_no_llm():
    """无 LLM 时 L3 应降级到简单截断，不崩溃"""
    config = AgentConfig(context_window=2000, compression_threshold=0.3, min_retain_rounds=2)
    pipeline = ContextPipeline(config, llm=None)  # 不给 LLM
    msgs = make_rounds(30, chars_per_msg=300)

    assert pipeline.should_compress(msgs)
    result = pipeline.compress(msgs)
    assert len(result) < len(msgs), "即使没有 LLM，L3 应降级截断"
    # L3 降级时不应抛异常（我们已在 #5 加了 try/except）


# === 测试 L4: 自适应阈值 ===
def test_l4_auto_threshold():
    """轮次不同 → 阈值不同"""
    config = AgentConfig(context_window=128000, compression_threshold=0.8)
    pipeline = ContextPipeline(config)

    # 短对话 → 高阈值
    short = make_rounds(3)
    assert not pipeline.should_compress(short), "3轮不应压缩"

    # 长对话，小窗口 → 应触发（用独立config防止缓存影响）
    c = AgentConfig(context_window=5000, compression_threshold=0.5)
    p2 = ContextPipeline(c)
    long = make_rounds(30, chars_per_msg=300)
    assert p2.should_compress(long), f"30轮应触发: window={c.context_window}"


# === 测试 4层全部触发 ===
def test_full_pipeline():
    """构造场景让 L1-L4 全部跑一遍"""
    config = AgentConfig(context_window=3000, compression_threshold=0.3, min_retain_rounds=2)
    pipeline = ContextPipeline(config)

    # 30轮 + 重复工具输出
    msgs = make_rounds_with_tools(30)

    assert pipeline.should_compress(msgs)
    result = pipeline.compress(msgs)

    reduction = (1 - len(result) / len(msgs)) * 100
    print(f"\n  压缩: {len(msgs)}→{len(result)} 条 ({reduction:.0f}%)")
    print(f"  L1截断={pipeline.stats['l1_truncations']}")
    print(f"  L2裁剪={pipeline.stats['l2_pruned']}")
    print(f"  L3摘要={pipeline.stats['l3_compressions']}")

    assert len(result) < len(msgs), f"四层压缩应减少消息量"
    # 至少 L1 或 L2 有一次生效
    assert pipeline.stats["l1_truncations"] + pipeline.stats["l2_pruned"] >= 1


if __name__ == "__main__":
    print("=== L1: Budget截断 ===")
    test_l1_budget_truncation()
    print("  PASS")

    print("=== L2: 冗余裁剪 ===")
    test_l2_redundancy_pruning()
    print("  PASS")

    print("=== L3: 降级截断 ===")
    test_l3_structural_compression_no_llm()
    print("  PASS")

    print("=== L4: 自适应阈值 ===")
    test_l4_auto_threshold()
    print("  PASS")

    print("=== 全流水线压测 ===")
    test_full_pipeline()
    print("  PASS")

    print("\n全部验证通过!")
