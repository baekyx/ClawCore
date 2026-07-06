"""
工具6: 网页抓取 — 带UA/超时/重试
"""

from typing import Dict, Any, List

import httpx
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


class WebFetchTool(Tool):
    """网页内容抓取"""

    def __init__(self):
        super().__init__(
            name="web_fetch",
            description="抓取指定URL的网页内容，提取纯文本。适合抓取新闻文章、文档等。提示：抓取新闻站首页来获取最新文章列表。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="url", type="string",
                          description="要抓取的URL。新闻类可抓 news.google.com 或具体新闻站",
                          required=True),
            ToolParameter(name="max_chars", type="integer",
                          description="最大返回字符数", required=False, default=5000),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        url = parameters.get("url", "")
        max_chars = min(parameters.get("max_chars", 5000), 20000)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        for attempt in range(2):  # 最多重试1次
            try:
                with httpx.Client(timeout=8, follow_redirects=True, headers=headers) as client:
                    resp = client.get(url)
                    resp.raise_for_status()

                html = resp.text

                # 提取纯文本
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
                        tag.decompose()
                    text = soup.get_text(separator='\n', strip=True)
                except ImportError:
                    import re
                    text = re.sub(r'<[^>]+>', ' ', html)
                    text = re.sub(r'\s+', ' ', text).strip()

                # 去空行
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                text = '\n'.join(lines)

                # 内容质量检测
                html_len = len(html)
                text_len = len(text)
                quality = text_len / max(html_len, 1)

                if quality < 0.05 or text_len < 200:
                    # 动态页面/JS渲染页面 → 告知Agent换策略
                    return ToolResponse.partial(
                        text=f"WARNING: {url} 是动态渲染页面，无法提取有效内容 (文本{text_len}字符, 质量{quality:.1%})。"
                             f"建议: 1) 搜索时加 'RSS' 关键词 2) 换用静态新闻源如 news.ycombinator.com 或 Google News",
                        data={"url": url, "dynamic": True, "quality": quality,
                              "text_length": text_len, "status": resp.status_code}
                    )

                if len(text) > max_chars:
                    text = text[:max_chars] + f"\n\n... (已截断，原文 {len(lines)} 行)"

                return ToolResponse.success(
                    text=f"{url}\n\n{text[:3000]}",
                    data={"url": url, "content": text, "quality": quality, "status": resp.status_code}
                )

            except httpx.HTTPStatusError as e:
                if attempt < 2:
                    continue
                return ToolResponse.error(code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"HTTP {e.response.status_code}")
            except httpx.TimeoutException:
                if attempt < 2:
                    continue
                return ToolResponse.error(code=ToolErrorCode.EXECUTION_ERROR,
                    message="请求超时")
            except Exception as e:
                if attempt < 2:
                    continue
                return ToolResponse.error(code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"抓取失败: {e}")
