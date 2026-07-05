"""
工具8: 任务管理器 — 声明式覆盖 + 单线程强制
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


@dataclass
class TodoItem:
    content: str
    status: str  # pending | in_progress | completed
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


@dataclass
class TodoList:
    summary: str = ""
    todos: List[TodoItem] = field(default_factory=list)

    def get_stats(self) -> dict:
        total = len(self.todos)
        completed = sum(1 for t in self.todos if t.status == "completed")
        in_progress = sum(1 for t in self.todos if t.status == "in_progress")
        return {"total": total, "completed": completed, "in_progress": in_progress,
                "pending": total - completed - in_progress}


class TaskManagerTool(Tool):
    """任务进度管理器"""

    def __init__(self):
        super().__init__(
            name="task_manager",
            description="管理任务列表。每次提交完整列表（声明式覆盖），最多1个in_progress任务。",
        )
        self.current_todos = TodoList()

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="action", type="string",
                          description="create|update|clear|status", required=True),
            ToolParameter(name="summary", type="string",
                          description="任务总体描述", required=False, default=""),
            ToolParameter(name="todos", type="array",
                          description='[{"content":"...", "status":"pending|in_progress|completed"}]',
                          required=False, default=[]),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        action = parameters.get("action", "status")

        if action == "clear":
            self.current_todos = TodoList()
            return ToolResponse.success(text="✅ 任务列表已清空")

        if action == "status":
            stats = self.current_todos.get_stats()
            if stats["total"] == 0:
                return ToolResponse.success(text="📋 暂无任务", data=stats)
            lines = [f"📋 [{stats['completed']}/{stats['total']}]"]
            for t in self.current_todos.todos:
                icon = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(t.status, "❓")
                lines.append(f"  {icon} {t.content}")
            return ToolResponse.success(text="\n".join(lines), data=stats)

        # create / update
        todos_data = parameters.get("todos", [])
        if isinstance(todos_data, str):
            try:
                todos_data = json.loads(todos_data)
            except json.JSONDecodeError:
                return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM,
                                           message="todos JSON 格式错误")

        # 单线程强制
        in_progress = sum(1 for t in todos_data if t.get("status") == "in_progress")
        if in_progress > 1:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM,
                                       message=f"最多1个in_progress，当前{in_progress}个")

        now = datetime.now().isoformat()
        todos = [
            TodoItem(content=t["content"], status=t["status"],
                     created_at=t.get("created_at", now), updated_at=now)
            for t in todos_data
        ]
        self.current_todos = TodoList(
            summary=parameters.get("summary", ""), todos=todos
        )

        stats = self.current_todos.get_stats()
        recap = f"📋 [{stats['completed']}/{stats['total']}] 任务已更新"
        return ToolResponse.success(text=recap, data=stats)
