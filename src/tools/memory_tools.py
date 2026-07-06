"""记忆工具 — Agent 可调用的记忆读写工具"""

from typing import Dict, Any, List

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


class MemoryWriteTool(Tool):
    """记忆写入工具 — Agent 用它主动记住信息"""

    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        super().__init__(
            name="memory_write",
            description="记住重要信息。当用户让你记住某事、告诉你偏好、或对话中有重要决策时调用。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="title", type="string",
                          description="简短标题，如'用户名'、'偏好语言'",
                          required=True),
            ToolParameter(name="content", type="string",
                          description="要记住的内容", required=True),
            ToolParameter(name="importance", type="number",
                          description="重要程度 0.0-1.0，用户明确说记住的=0.9",
                          required=False, default=0.7),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        if not self.memory_manager:
            return ToolResponse.error(code=ToolErrorCode.INTERNAL_ERROR,
                                       message="记忆管理器未初始化")

        title = parameters.get("title", "")
        content = parameters.get("content", "")
        importance = parameters.get("importance", 0.7)

        self.memory_manager.remember_fact(title, content, importance=importance)
        self.memory_manager.remember_user_preference(
            title.lower().replace(" ", "_"),
            content
        )

        return ToolResponse.success(
            text=f"已记住: {title}",
            data={"title": title, "stored": True}
        )


class MemoryRecallTool(Tool):
    """记忆召回工具 — Agent 用来回忆之前的对话"""

    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        super().__init__(
            name="memory_recall",
            description="回忆之前的对话和已知信息。当你需要确认是否了解用户、查找之前的偏好时使用。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="query", type="string",
                          description="要回忆的关键词或问题", required=True),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        if not self.memory_manager:
            return ToolResponse.error(code=ToolErrorCode.INTERNAL_ERROR,
                                       message="记忆管理器未初始化")

        query = parameters.get("query", "")

        # 先查工作记忆
        results = []
        if self.memory_manager.working:
            wm = self.memory_manager.working.search(keyword=query)
            for item in wm:
                results.append(f"- [{item['namespace']}] {item['title']}: {item['content'][:200]}")

        # 再查长期记忆
        if self.memory_manager.long_term and self.memory_manager.long_term.count() > 0:
            ltm = self.memory_manager.long_term.hybrid_search(query, top_k=3)
            for item in ltm:
                results.append(f"- [长期] {item['content'][:200]}")

        if not results:
            return ToolResponse.success(text=f"未找到关于'{query}'的记忆", data={"results": []})

        text = f"关于'{query}'的记忆:\n" + "\n".join(results)
        return ToolResponse.success(text=text, data={"results": results})
