"""
工具5: 网络搜索 — 多引擎链 (DDG → Bing中国 → 搜狗 → 百度)
中国IP下自动降级到国内搜索引擎
"""

import re
import time
from typing import Dict, Any, List, Optional
from urllib.parse import quote

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


# ── 通用请求头 ──

_HEADERS_DESKTOP = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

_HEADERS_MOBILE = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


# ── 搜索引擎定义 ──

class SearchEngine:
    """搜索引擎抽象"""

    def __init__(self, name: str, url_template: str, headers: Dict = None,
                 timeout: float = 8.0, retries: int = 1):
        self.name = name
        self.url_template = url_template
        self.headers = headers or _HEADERS_DESKTOP
        self.timeout = timeout
        self.retries = retries

    def search(self, query: str, max_results: int) -> Optional[List[Dict]]:
        raise NotImplementedError


class DuckDuckGoEngine(SearchEngine):
    """DDG — 优先尝试，速度快"""

    def __init__(self):
        super().__init__("DuckDuckGo", "", timeout=5.0)

    def search(self, query: str, max_results: int) -> Optional[List[Dict]]:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:300],
                    })
            return results if results else None
        except Exception:
            return None


class BingCNEngine(SearchEngine):
    """必应中国 — 国内直连，国际内容"""

    def __init__(self):
        super().__init__("BingCN", "https://cn.bing.com/search?q={query}&count={count}",
                         _HEADERS_DESKTOP, timeout=8.0, retries=2)

    def search(self, query: str, max_results: int) -> Optional[List[Dict]]:
        import httpx
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True,
                              headers=self.headers) as client:
                resp = client.get(self.url_template.format(query=quote(query), count=max_results))
                if resp.status_code != 200:
                    return None
            return self._parse(resp.text, max_results)
        except Exception:
            return None

    def _parse(self, html: str, limit: int) -> List[Dict]:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            for item in soup.select('li.b_algo'):
                title_el = item.select_one('h2 a')
                snippet_el = item.select_one('.b_caption p')
                if title_el:
                    results.append({
                        "title": title_el.get_text(strip=True),
                        "url": title_el.get('href', ''),
                        "snippet": snippet_el.get_text(strip=True)[:300] if snippet_el else '',
                    })
                if len(results) >= limit:
                    break
            return results
        except Exception:
            return []


class SogouEngine(SearchEngine):
    """搜狗 — 反爬弱，国内友好"""

    def __init__(self):
        super().__init__("Sogou", "https://www.sogou.com/web?query={query}", _HEADERS_DESKTOP,
                         timeout=8.0, retries=1)

    def search(self, query: str, max_results: int) -> Optional[List[Dict]]:
        import httpx
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True,
                              headers=self.headers) as client:
                resp = client.get(self.url_template.format(query=quote(query)))
                if resp.status_code != 200:
                    return None
            return self._parse(resp.text, max_results)
        except Exception:
            return None

    def _parse(self, html: str, limit: int) -> List[Dict]:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            for item in soup.select('.rb, .vrwrap, .results > div'):
                title_el = item.select_one('h3 a, .vr-title a')
                snippet_el = item.select_one('.str-text, .star-wiki, p')
                if title_el:
                    results.append({
                        "title": title_el.get_text(strip=True),
                        "url": title_el.get('href', ''),
                        "snippet": snippet_el.get_text(strip=True)[:300] if snippet_el else '',
                    })
                if len(results) >= limit:
                    break
            return results
        except Exception:
            return []


# ── 多引擎搜索工具 ──

_ENGINES = [DuckDuckGoEngine(), BingCNEngine(), SogouEngine()]


class WebSearchTool(Tool):
    """多引擎网络搜索 — DDG → Bing中国 → 搜狗 自动降级"""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="搜索网络获取信息。自动切换搜索引擎。提示：搜索新闻时关键词包含日期（如'2026年7月AI新闻'），"
                        "优先使用具体词而非宽泛词。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="query", type="string",
                          description="搜索关键词。新闻类请带日期", required=True),
            ToolParameter(name="max_results", type="integer",
                          description="最大结果数(1-10)", required=False, default=5),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        query = parameters.get("query", "")
        max_results = min(parameters.get("max_results", 5), 10)

        if not query.strip():
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="query 不能为空")

        errors = []

        for engine in _ENGINES:
            for attempt in range(engine.retries + 1):
                try:
                    results = engine.search(query, max_results)
                    if results:
                        text = f"[{engine.name}] 搜索 '{query}':\n"
                        for i, r in enumerate(results[:max_results], 1):
                            text += f"{i}. {r['title']}\n   {r['snippet']}\n   {r['url']}\n"
                        return ToolResponse.success(
                            text=text,
                            data={"engine": engine.name, "query": query, "results": results[:max_results]}
                        )
                except Exception as e:
                    errors.append(f"{engine.name}: {e}")
                    if attempt < engine.retries:
                        time.sleep(0.5)
            time.sleep(0.3)  # 引擎间短暂间隔

        # 全部引擎失败
        return ToolResponse.error(
            code=ToolErrorCode.EXECUTION_ERROR,
            message=f"所有搜索引擎不可用 (DDG/BingCN/Sogou)。建议用 web_fetch 直接抓取已知网站。"
        )
