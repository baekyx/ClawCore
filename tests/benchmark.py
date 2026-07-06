"""
ClawCore Benchmark — 评估体系
  Part A: 检索指标 (HitRate@K, MRR@K)
  Part B: 压缩率实测
  Part C: 端到端正确率
"""
import sys
import math
import time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_config, AgentConfig
from src.memory import LongTermMemory
from src.context import ContextPipeline


# ═══════════════════════════════════════════
# Part A: 检索指标
# ═══════════════════════════════════════════

# 测试记忆语料（20条）
TEST_MEMORIES = [
    {"content": "用户在2024年3月购买了深圳南山区的房产，总价850万", "type": "user_fact"},
    {"content": "用户偏好使用Python进行后端开发，不喜欢Java", "type": "user_pref"},
    {"content": "用户每天早上8点到公司，晚上7点离开", "type": "user_fact"},
    {"content": "上次讨论决定了使用PostgreSQL替代MySQL作为主数据库", "type": "decision"},
    {"content": "用户父亲今年65岁，患有高血压，需要定期体检", "type": "user_fact"},
    {"content": "公司2025年Q2营收同比增长23%，主要来自海外市场", "type": "business"},
    {"content": "用户计划2026年9月去日本旅游，预算3万元", "type": "user_plan"},
    {"content": "团队决定将微服务从12个合并为5个核心服务", "type": "decision"},
    {"content": "用户喜欢喝咖啡，尤其偏爱埃塞俄比亚耶加雪菲", "type": "user_pref"},
    {"content": "上次故障是Redis集群内存溢出导致，已扩容到32GB", "type": "incident"},
    {"content": "用户的孩子小明今年8岁，在学钢琴和游泳", "type": "user_fact"},
    {"content": "竞品分析显示主要对手的定价比我们低15%", "type": "business"},
    {"content": "用户常用的开发工具是VS Code和JetBrains全家桶", "type": "user_pref"},
    {"content": "公司计划2026年Q4启动AIGC相关新业务线", "type": "business"},
    {"content": "用户住址是北京市朝阳区望京街道，离公司3公里", "type": "user_fact"},
    {"content": "上次代码审查发现3个SQL注入漏洞，已全部修复", "type": "incident"},
    {"content": "用户配偶是医生，在北京协和医院工作", "type": "user_fact"},
    {"content": "系统QPS从年初的500提升到了现在的2000", "type": "business"},
    {"content": "用户不喜欢吃肉，偏好素食和海鲜", "type": "user_pref"},
    {"content": "上次团建去了古北水镇，大家反馈很好", "type": "social"},
]

# 测试查询 + 正确答案 ID（人工标注）
TEST_QUERIES = [
    {"query": "用户的父亲有什么健康问题？", "relevant_ids": [4]},
    {"query": "用户住在哪里？", "relevant_ids": [0, 14]},
    {"query": "用户喜欢喝什么？", "relevant_ids": [8]},
    {"query": "上次故障是什么原因？", "relevant_ids": [9]},
    {"query": "用户的开发工具偏好？", "relevant_ids": [1, 12]},
    {"query": "公司的业务增长情况？", "relevant_ids": [5, 17]},
    {"query": "用户的家庭情况？", "relevant_ids": [4, 10, 16]},
    {"query": "用户的旅游计划？", "relevant_ids": [6]},
    {"query": "重要技术决策有哪些？", "relevant_ids": [3, 7]},
    {"query": "用户不喜欢什么？", "relevant_ids": [1, 18]},
]


def load_test_data(ltm):
    """加载测试语料到 L3"""
    ltm.add_batch(TEST_MEMORIES)
    print(f"  Loaded {ltm.count()} memories")


def hit_rate_at_k(results, relevant_ids, k):
    """HitRate@K: 前K个结果中至少有一个相关的比例"""
    top_k_ids = {r["id"] for r in results[:k]}
    return 1 if top_k_ids & set(relevant_ids) else 0


def reciprocal_rank(results, relevant_ids):
    """MRR: 第一个相关结果的倒数排名"""
    for rank, r in enumerate(results, 1):
        if r["id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


def run_retrieval_benchmark(ltm):
    print("\n=== Part A: Retrieval Benchmark ===")
    results = {"hitrate@1": [], "hitrate@3": [], "hitrate@5": [], "hitrate@10": [], "mrr": []}

    for q in TEST_QUERIES:
        retrieved = ltm.hybrid_search(q["query"], top_k=10)

        results["hitrate@1"].append(hit_rate_at_k(retrieved, q["relevant_ids"], 1))
        results["hitrate@3"].append(hit_rate_at_k(retrieved, q["relevant_ids"], 3))
        results["hitrate@5"].append(hit_rate_at_k(retrieved, q["relevant_ids"], 5))
        results["hitrate@10"].append(hit_rate_at_k(retrieved, q["relevant_ids"], 10))
        results["mrr"].append(reciprocal_rank(retrieved, q["relevant_ids"]))

    # Debug: 单条查询详情
    sample = TEST_QUERIES[0]
    sample_results = ltm.hybrid_search(sample["query"], top_k=5)
    print(f"\n  Debug: '{sample['query']}' → expect IDs {sample['relevant_ids']}")
    for r in sample_results:
        print(f"    id={r['id']} score={r['score']:.3f} type={r['type']} content={r['content'][:50]}...")

    print(f"\n  Queries: {len(TEST_QUERIES)}")
    for metric, values in results.items():
        avg = sum(values) / len(values)
        print(f"  {metric.upper():>12s}: {avg:.4f}  ({sum(values)}/{len(values)})")

    return results


# ═══════════════════════════════════════════
# Part B: 压缩率
# ═══════════════════════════════════════════

def make_long_conversation(rounds: int, chars=300):
    """生成 N 轮模拟对话"""
    msgs = [{"role": "system", "content": "你是一个AI助手"}]
    for i in range(rounds):
        msgs.append({"role": "user", "content": f"问题{i}: " + "A" * chars})
        msgs.append({"role": "assistant", "content": f"回答{i}: " + "B" * chars})
    return msgs


def run_compression_benchmark():
    print("\n=== Part B: Compression Benchmark ===")
    config = AgentConfig(context_window=8000, compression_threshold=0.5, min_retain_rounds=3)
    pipeline = ContextPipeline(config)

    scenarios = [
        {"name": "短对话(10轮)", "rounds": 10, "chars": 200},
        {"name": "中对话(30轮)", "rounds": 30, "chars": 200},
        {"name": "长对话(50轮)", "rounds": 50, "chars": 300},
        {"name": "超长对话(100轮)", "rounds": 100, "chars": 200},
    ]

    for s in scenarios:
        msgs = make_long_conversation(s["rounds"], s["chars"])
        original = len(msgs)
        original_tokens = sum(len(m["content"]) for m in msgs) // 4

        if pipeline.should_compress(msgs):
            compressed = pipeline.compress(msgs)
            after = len(compressed)
            after_tokens = sum(len(m["content"]) for m in compressed) // 4
            reduction = (1 - after_tokens / max(original_tokens, 1)) * 100
        else:
            after = original
            after_tokens = original_tokens
            reduction = 0

        print(f"  {s['name']:>15s}: {original}→{after}条 | "
              f"{original_tokens}→{after_tokens} tokens | "
              f"-{reduction:.0f}% | "
              f"L1={pipeline.stats['l1_truncations']} "
              f"L2={pipeline.stats['l2_pruned']} "
              f"L3={pipeline.stats['l3_compressions']}")


# ═══════════════════════════════════════════
# Part C: 端到端正确率
# ═══════════════════════════════════════════

E2E_TESTS = [
    {"query": "1+2*3等于几", "expected": ["7"]},
    {"query": "sqrt(144)是多少", "expected": ["12"]},
    {"query": "hello的英文是什么意思", "expected": ["你好", "您好"]},
    {"query": "中国首都是哪个城市", "expected": ["北京"]},
    {"query": "水的化学式是什么", "expected": ["H2O", "H₂O"]},
    {"query": "2024年奥运会在哪个城市举办", "expected": ["巴黎", "Paris"]},
]


def run_e2e_benchmark(agent):
    print("\n=== Part C: End-to-End Accuracy ===")
    correct = 0
    total = len(E2E_TESTS)

    for t in E2E_TESTS:
        result = agent.run(t["query"])
        passed = any(exp in result for exp in t["expected"])
        status = "PASS" if passed else "FAIL"
        if passed:
            correct += 1
        print(f"  [{status}] {t['query']:>25s}  expected={t['expected']}  got={result[:40]}...")

    acc = correct / total * 100
    print(f"\n  Accuracy: {correct}/{total} ({acc:.0f}%)")
    return acc


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    config = get_config()

    # Part A: Retrieval (needs Postgres + BGE-M3)
    print("=" * 60)
    print("ClawCore Benchmark")
    print("=" * 60)
    try:
        ltm = LongTermMemory(config.postgres, config.embedding, config.retrieval)
        # 清空旧数据 + 加载 benchmark 专属数据
        with ltm._cursor() as cur:
            cur.execute(f"DELETE FROM {config.postgres.vector_table}")
        ltm._bm25_texts = []
        ltm._bm25_metadata = []
        ltm._bm25 = None
        load_test_data(ltm)
        run_retrieval_benchmark(ltm)
    except Exception as e:
        print(f"\n  [SKIP] Retrieval benchmark: {e}")

    # Part B: Compression (no deps)
    run_compression_benchmark()

    # Part C: E2E (needs LLM API)
    print("\n=== Part C: End-to-End ===")
    try:
        from src.llm import create_llm
        from src.agent_loop.react_loop import ClawCoreAgent
        from src.tools import create_default_registry
        llm = create_llm(config)
        agent = ClawCoreAgent(name="Benchmark", llm=llm, config=config,
            tool_registry=create_default_registry(), max_steps=3)
        run_e2e_benchmark(agent)
    except Exception as e:
        print(f"  [SKIP] E2E benchmark: {e}")

    print("\n" + "=" * 60)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
