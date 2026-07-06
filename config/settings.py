"""ClawCore 统一配置管理 — 参考 FinFlow dataclass 风格 + HelloAgents Config 体系"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """LLM 配置"""
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


@dataclass
class PostgresConfig:
    """Postgres 配置 — 工作记忆(Layer2) + 长期记忆(Layer3 pgvector)"""
    host: str = "localhost"
    port: int = 5432
    database: str = "clawcore"
    user: str = "postgres"
    password: str = ""
    # pgvector 扩展
    vector_dim: int = 1024  # BGE-M3 维度
    vector_table: str = "long_term_memories"


@dataclass
class EmbeddingConfig:
    """向量化配置"""
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    normalize: bool = True


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = 10
    rerank_top_k: int = 3
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60
    min_relevance: float = 0.3


@dataclass
class AgentConfig:
    """Agent 配置"""
    max_steps: int = 10
    context_window: int = 128000
    compression_threshold: float = 0.8
    min_retain_rounds: int = 10
    tool_timeout: int = 30
    max_concurrent_tools: int = 3


@dataclass
class SkillConfig:
    """Skill 配置"""
    skills_dir: str = "data/skill_artifacts"
    auto_register: bool = True
    max_skills_in_prompt: int = 5
    max_tokens_per_skill_meta: int = 100


@dataclass
class ClawCoreConfig:
    """ClawCore 全局配置 — 单例模式"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    debug: bool = False


_config: Optional[ClawCoreConfig] = None


def get_config() -> ClawCoreConfig:
    """获取全局配置单例，首次调用时从环境变量加载"""
    global _config
    if _config is None:
        load_dotenv()
        _config = ClawCoreConfig(
            llm=LLMConfig(
                model=os.getenv("LLM_MODEL_ID", "deepseek-chat"),
                api_key=os.getenv("LLM_API_KEY", ""),
                base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            ),
            postgres=PostgresConfig(
                host=os.getenv("PG_HOST", "localhost"),
                port=int(os.getenv("PG_PORT", "5432")),
                database=os.getenv("PG_DATABASE", "clawcore"),
                user=os.getenv("PG_USER", "postgres"),
                password=os.getenv("PG_PASSWORD", ""),
            ),
            embedding=EmbeddingConfig(
                model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
                device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            ),
            agent=AgentConfig(
                max_steps=int(os.getenv("MAX_STEPS", "10")),
            ),
        )
    return _config
