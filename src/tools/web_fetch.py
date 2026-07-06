"""
工具6: 网页抓取工具
"""

from typing import Dict, Any, List

import httpx
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


class WebFetchTool(Tool):
    """网页内容抓取 — 获取网页文本内容"""

    def __init__(self):
        super().__init__(
            name="web_fetch",
            description="抓取指定 URL 的网页内容，提取纯文本。适合阅读文章、文档等。",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="url", type="string",
                          description="要抓取的网页 URL", required=True),
            ToolParameter(name="max_chars", type="integer",
                          description="最大返回字符数", required=False, default=5000),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        url = parameters.get("url", "")
        max_chars = min(parameters.get("max_chars", 5000), 20000)

        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ClawCore/1.0)"
                })
                resp.raise_for_status()

            # 简单提取文本
            html = resp.text

            # 尝试用 BeautifulSoup 提取纯文本
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                # 移除 script/style
                for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                    tag.decompose()
                text = soup.get_text(separator='\n', strip=True)
            except ImportError:
                # 降级：简单去掉 HTML 标签
                import re
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'\s+', ' ', text)

            # 截断
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n... (已截断，原文共 {len(text)} 字符)"

            return ToolResponse.success(
                text=f"抓取 {url}:\n\n{text[:3000]}",
                data={"url": url, "content": text, "status_code": resp.status_code}
            )

        except httpx.HTTPStatusError as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"HTTP {e.response.status_code}: {url}"
            )
        except httpx.TimeoutException:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"请求超时: {url}"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"抓取失败: {e}"
            )
