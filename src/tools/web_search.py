"""
工具5: 网络搜索工具 — DuckDuckGo 搜索
"""

from typing import Dict, Any, List

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


class WebSearchTool(Tool):
    """网络搜索 — 使用 DuckDuckGo"""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="搜索网络获取最新信息。返回标题、摘要和链接。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="query", type="string",
                          description="搜索关键词", required=True),
            ToolParameter(name="max_results", type="integer",
                          description="最大结果数", required=False, default=5),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        query = parameters.get("query", "")
        max_results = min(parameters.get("max_results", 5), 10)

        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:300],
                    })

            if not results:
                return ToolResponse.success(
                    text=f"未找到 '{query}' 的相关结果",
                    data={"query": query, "results": []}
                )

            text_lines = [f"搜索 '{query}' 的结果:"] + [
                f"{i+1}. {r['title']}\n   {r['snippet']}\n   {r['url']}"
                for i, r in enumerate(results)
            ]

            return ToolResponse.success(
                text="\n\n".join(text_lines),
                data={"query": query, "results": results}
            )

        except ImportError:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message="请安装 duckduckgo-search: pip install duckduckgo-search"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"搜索失败: {e}"
            )
