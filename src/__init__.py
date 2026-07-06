"""ClawCore — 基于多层记忆与 Skill 自进化的通用 Agent 助手"""

__version__ = "0.1.0"

from .agent_loop.react_loop import ClawCoreAgent
from .tools import create_default_registry
from .llm import create_llm

__all__ = ["ClawCoreAgent", "create_default_registry", "create_llm"]
