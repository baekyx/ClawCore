"""ClawCore LLM 适配层 — 直接包装 HelloAgents 的 HelloAgentsLLM"""

import sys
from pathlib import Path

# 导入 HelloAgents（开发阶段直接引用本地路径）
HELLO_AGENTS_PATH = Path("D:/code/HelloAgents")
if str(HELLO_AGENTS_PATH) not in sys.path:
    sys.path.insert(0, str(HELLO_AGENTS_PATH))

from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.llm_response import LLMResponse, LLMToolResponse, ToolCall, StreamStats
from hello_agents.core.config import Config as HelloAgentsConfig

from config.settings import ClawCoreConfig, get_config


def create_llm(config: ClawCoreConfig = None) -> HelloAgentsLLM:
    """创建 LLM 实例，自动从环境变量读取配置"""
    if config is None:
        config = get_config()

    return HelloAgentsLLM(
        model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout=config.llm.timeout,
    )


__all__ = [
    "HelloAgentsLLM",
    "LLMResponse",
    "LLMToolResponse",
    "ToolCall",
    "StreamStats",
    "HelloAgentsConfig",
    "create_llm",
]
