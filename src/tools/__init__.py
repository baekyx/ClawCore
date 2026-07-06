"""ClawCore 工具系统 — 9 个核心工具"""
import sys
from pathlib import Path

HELLO_AGENTS_PATH = Path("D:/code/HelloAgents")
if str(HELLO_AGENTS_PATH) not in sys.path:
    sys.path.insert(0, str(HELLO_AGENTS_PATH))

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse, ToolStatus
from hello_agents.tools.registry import ToolRegistry
from hello_agents.tools.errors import ToolErrorCode

from .file_ops import FileReadTool, FileWriteTool, FileEditTool
from .calculator import CalculatorTool
from .web_search import WebSearchTool
from .web_fetch import WebFetchTool
from .finish import FinishTool
from .task_manager import TaskManagerTool
from .skill_invoke import SkillInvokeTool
from .memory_tools import MemoryWriteTool, MemoryRecallTool


def create_default_registry(skill_manager=None, memory_manager=None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(FileReadTool())
    registry.register_tool(FileWriteTool())
    registry.register_tool(FileEditTool())
    registry.register_tool(CalculatorTool())
    registry.register_tool(WebSearchTool())
    registry.register_tool(WebFetchTool())
    registry.register_tool(FinishTool())
    registry.register_tool(TaskManagerTool())
    registry.register_tool(SkillInvokeTool(skill_manager=skill_manager))
    registry.register_tool(MemoryWriteTool(memory_manager=memory_manager))
    registry.register_tool(MemoryRecallTool(memory_manager=memory_manager))
    return registry


__all__ = [
    "Tool", "ToolParameter", "ToolResponse", "ToolStatus",
    "ToolRegistry", "ToolErrorCode",
    "FileReadTool", "FileWriteTool", "FileEditTool",
    "CalculatorTool", "WebSearchTool", "WebFetchTool",
    "FinishTool", "TaskManagerTool", "SkillInvokeTool",
    "MemoryWriteTool", "MemoryRecallTool",
    "create_default_registry",
]
