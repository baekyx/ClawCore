"""测试 L3 长期记忆"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_config
from src.memory import LongTermMemory

c = get_config()
ltm = LongTermMemory(c.postgres, c.embedding, c.retrieval)
print(f"L3 init OK: {ltm.count()} 条记录")

# 写入测试数据
ltm.add("张三住在北京朝阳区，喜欢喝咖啡", memory_type="user_fact", importance=0.9)
ltm.add("ClawCore是多层记忆Agent框架，支持Skill自进化", memory_type="knowledge", importance=0.8)
ltm.add("张三上次问过深圳科技园房价，预算800万", memory_type="conversation", importance=0.7)
print(f"写入 3 条, 共 {ltm.count()} 条")

# 混合检索
print("\n--- 检索: 张三住在哪 ---")
for r in ltm.hybrid_search("张三住在哪", top_k=3):
    print(f"  [{r['type']}] {r['content'][:60]} (score={r['score']:.3f})")

print("\n--- 检索: ClawCore是什么 ---")
for r in ltm.hybrid_search("ClawCore是什么", top_k=3):
    print(f"  [{r['type']}] {r['content'][:60]} (score={r['score']:.3f})")

print("\nL3 测试完成!")
