"""
工具7: Finish 工具 — Agent 完成信号
"""

from typing import Dict, Any, List

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse


class FinishTool(Tool):
    """完成信号工具 — Agent 调用此工具宣告任务结束"""

    def __init__(self):
        super().__init__(
            name="finish",
            description="当任务完成、有足够信息给出最终答案时调用。参数 answer 为最终回答。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="answer", type="string",
                          description="最终答案，完整、准确、直接回答用户问题",
                          required=True),
            ToolParameter(name="summary", type="string",
                          description="任务执行摘要（可选）", required=False, default=""),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        answer = parameters.get("answer", "")
        summary = parameters.get("summary", "")

        parts = []
        if summary:
            parts.append(f"📋 摘要: {summary}")
        parts.append(f"✅ 答案: {answer}")

        return ToolResponse.success(
            text="\n".join(parts),
            data={"answer": answer, "summary": summary, "finished": True}
        )
