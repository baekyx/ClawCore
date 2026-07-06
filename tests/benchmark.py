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

# 测试查询 + 正确答案关键词（内容匹配，不依赖PG自增ID）
TEST_QUERIES = [
    {"query": "用户的父亲有什么健康问题？", "keywords": ["父亲", "高血压"]},
    {"query": "用户住在哪里？", "keywords": ["朝阳区", "望京"]},
    {"query": "用户喜欢喝什么？", "keywords": ["咖啡", "耶加雪菲"]},
    {"query": "上次故障是什么原因？", "keywords": ["Redis", "内存溢出"]},
    {"query": "用户的开发工具偏好？", "keywords": ["VS Code", "JetBrains"]},
    {"query": "公司的业务增长情况？", "keywords": ["营收", "QPS"]},
    {"query": "用户的家庭情况？", "keywords": ["父亲", "孩子", "医生"]},
    {"query": "用户的旅游计划？", "keywords": ["日本", "旅游"]},
    {"query": "重要技术决策有哪些？", "keywords": ["PostgreSQL", "微服务"]},
    {"query": "用户不喜欢什么？", "keywords": ["不喜欢", "素食"]},
]


def load_test_data(ltm):
    """加载测试语料到 L3，返回每条记忆的实际 PG ID"""
    # 重置自增 ID
    with ltm._cursor() as cur:
        cur.execute("ALTER SEQUENCE long_term_memories_id_seq RESTART WITH 1")
    ltm._bm25_texts = []
    ltm._bm25_metadata = []
    ltm._bm25 = None

    ids = ltm.add_batch(TEST_MEMORIES)
    print(f"  Loaded {len(ids)} memories (IDs: {ids[0]}-{ids[-1]})")
    return ids


def is_relevant(content, keywords):
    """内容匹配：至少命中一个关键词"""
    return any(kw in content for kw in keywords)


def hit_rate_at_k(results, keywords, k):
    """HitRate@K: 前K个结果中至少有一个相关"""
    for r in results[:k]:
        if is_relevant(r["content"], keywords):
            return 1
    return 0


def reciprocal_rank(results, keywords):
    """MRR: 第一个相关结果的倒数排名"""
    for rank, r in enumerate(results, 1):
        if is_relevant(r["content"], keywords):
            return 1.0 / rank
    return 0.0


def run_retrieval_benchmark(ltm):
    print("\n=== Part A: Retrieval Benchmark ===")
    results = {"hitrate@1": [], "hitrate@3": [], "hitrate@5": [], "hitrate@10": [], "mrr": []}

    for q in TEST_QUERIES:
        retrieved = ltm.hybrid_search(q["query"], top_k=10)

        results["hitrate@1"].append(hit_rate_at_k(retrieved, q["keywords"], 1))
        results["hitrate@3"].append(hit_rate_at_k(retrieved, q["keywords"], 3))
        results["hitrate@5"].append(hit_rate_at_k(retrieved, q["keywords"], 5))
        results["hitrate@10"].append(hit_rate_at_k(retrieved, q["keywords"], 10))
        results["mrr"].append(reciprocal_rank(retrieved, q["keywords"]))

    # Debug: 单条查询详情
    sample = TEST_QUERIES[0]
    sample_results = ltm.hybrid_search(sample["query"], top_k=5)
    print(f"\n  Debug: '{sample['query']}' → expect keywords {sample['keywords']}")
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
    # Tier 1: 简单知识
    {"query": "1+2*3等于几", "expected": ["7"], "tier": 1, "category": "算术"},
    {"query": "中国首都是哪个城市", "expected": ["北京"], "tier": 1, "category": "常识"},

    # Tier 2: 需要工具
    {"query": "sqrt(144)是多少", "expected": ["12"], "tier": 2, "category": "工具-计算"},
    {"query": "帮我算一下 15*8+12/3", "expected": ["124"], "tier": 2, "category": "工具-计算"},
    {"query": "搜索并告诉我Python最新版本是什么", "expected": ["Python", "3."], "tier": 2, "category": "工具-搜索"},

    # Tier 3: 多步推理+记忆
    {"query": "你还记得我叫什么名字吗", "expected": ["张三", "名字", "没有告诉"], "tier": 3, "category": "记忆召回"},
    {"query": "我之前和你说过我住在哪里", "expected": ["北京", "没有说"], "tier": 3, "category": "记忆召回"},

    # Tier 4: 复杂推理
    {"query": "已知函数f(x)=x²-4x+3, x∈[0,5]，求最大值和最小值", "expected": ["3", "-1", "8"], "tier": 4, "category": "数学推理"},
    {"query": "一个班级有40人，男生比女生多4人，男生多少人", "expected": ["22"], "tier": 4, "category": "数学推理"},

    # Tier 5: 综合能力
    {"query": "分析sample_notes.txt这个文件讲了什么内容", "expected": ["AI", "Agent", "多Agent", "记忆", "观察"], "tier": 5, "category": "文件+理解"},
    {"query": "用一句话介绍Python是什么", "expected": ["Python", "编程", "语言"], "tier": 5, "category": "综合"},

    # Edge: 错误处理
    {"query": "帮我读取一个不存在的文件 /nonexistent.txt", "expected": [], "tier": "edge", "category": "错误处理"},
    {"query": "3/0等于多少", "expected": ["不能", "错误", "0不能"], "tier": "edge", "category": "错误处理"},
]


def run_e2e_benchmark(agent):
    print("\n=== Part C: End-to-End Accuracy ===\n")

    by_tier = {}
    correct = 0
    total = len(E2E_TESTS)

    for t in E2E_TESTS:
        try:
            result = agent.run(t["query"])
        except Exception as e:
            result = str(e)

        tier = str(t["tier"])
        if tier not in by_tier:
            by_tier[tier] = {"total": 0, "correct": 0}

        by_tier[tier]["total"] += 1

        # Edge 用例：期望失败/拒绝的响应
        if t.get("category") == "错误处理":
            passed = (
                "错误" in result or "[ERR]" in result or
                "不存在" in result or "不能" in result
            )
        else:
            passed = any(exp in result for exp in t["expected"])

        if passed:
            correct += 1
            by_tier[tier]["correct"] += 1

        status = "PASS" if passed else "FAIL"
        tier_str = str(t['tier'])
        print(f"  [{status}] T{tier_str:<4s} {t['category']:<8s} | "
              f"{t['query'][:45]:<45s} | {result[:50]}...")

    # 按难度统计
    print(f"\n  {'Tier':<8s} {'Category':<12s} {'Accuracy'}")
    print(f"  {'-'*35}")
    for tier in sorted(by_tier.keys(), key=lambda x: (x.isdigit(), x)):
        d = by_tier[tier]
        pct = d["correct"] / d["total"] * 100
        tier_name = {"1":"Basic","2":"Tool","3":"Memory","4":"Reason","5":"Complex","edge":"Edge"}.get(tier, tier)
        print(f"  T{tier:<7s} {tier_name:<12s} {d['correct']}/{d['total']} ({pct:.0f}%)")

    acc = correct / total * 100
    print(f"\n  Total: {correct}/{total} ({acc:.0f}%)")
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
        from src.memory import MemoryManager
        llm = create_llm(config)
        memory = MemoryManager(config)
        agent = ClawCoreAgent(name="Benchmark", llm=llm, config=config,
            tool_registry=create_default_registry(memory_manager=memory),
            memory_manager=memory, max_steps=5)
        run_e2e_benchmark(agent)
    except Exception as e:
        print(f"  [SKIP] E2E benchmark: {e}")

    print("\n" + "=" * 60)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
