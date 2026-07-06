"""
工具5: 网络搜索 — DuckDuckGo + 备用源
"""

from typing import Dict, Any, List
import time

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


class WebSearchTool(Tool):
    """网络搜索"""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="搜索网络获取信息。提示：搜索新闻类问题时，关键词应包含日期（如'2026年7月 AI新闻'），优先使用具体关键词而非宽泛词。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="query", type="string",
                          description="搜索关键词。搜索新闻请带日期如'2026-07-06 AI新闻'",
                          required=True),
            ToolParameter(name="max_results", type="integer",
                          description="最大结果数(1-10)", required=False, default=5),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        query = parameters.get("query", "")
        max_results = min(parameters.get("max_results", 5), 10)

        # 尝试 ddgs
        result = self._try_ddgs(query, max_results)
        if result:
            return result

        # 备用：直接返回引导 Agent 去抓取新闻源
        return ToolResponse.success(
            text=f"DuckDuckGo 搜索 '{query}' 暂时不可用。建议：\n"
                 f"1. 直接用 web_fetch 抓取 news.google.com 或 theverge.com/ai\n"
                 f"2. 换更具体的关键词重试",
            data={"query": query, "results": [], "fallback": True}
        )

    def _try_ddgs(self, query: str, max_results: int):
        for attempt in range(2):
            try:
                # 先试新版 ddgs
                try:
                    from ddgs import DDGS
                except ImportError:
                    from duckduckgo_search import DDGS

                results = []
                with DDGS(headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }) as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", "")[:300],
                        })

                if results:
                    text = f"搜索 '{query}' 的结果:\n"
                    for i, r in enumerate(results, 1):
                        text += f"{i}. {r['title']}\n   {r['snippet']}\n   {r['url']}\n\n"
                    return ToolResponse.success(text=text, data={"query": query, "results": results})

                return None  # 无结果，非错误

            except Exception as e:
                if attempt == 0:
                    time.sleep(1)
                    continue
                # 最后一次尝试失败则不抛异常，返回 None
                return None

        return None
