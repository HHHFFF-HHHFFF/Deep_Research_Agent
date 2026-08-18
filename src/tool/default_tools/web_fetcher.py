"""提供web fetcher相关实现。"""

from typing import Any

from pydantic import Field

from src.logger import logger
from src.registry import TOOL
from src.tool.types import Tool, ToolExtra, ToolResponse
from src.utils import fetch_url

_WEB_FETCHER_DESCRIPTION = """Visit a webpage at a given URL and return its text content.
Use this tool to fetch and read content from web pages.
The tool will return the page title and markdown-formatted content.

Args:
- url (str): The URL of the webpage to fetch.

Example: {"name": "web_fetcher", "args": {"url": "https://www.google.com"}}.
"""


@TOOL.register_module(force=True)
class WebFetcherTool(Tool):
    """定义 `WebFetcherTool`，封装相关数据与行为。"""

    name: str = "web_fetcher"
    description: str = _WEB_FETCHER_DESCRIPTION
    metadata: dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(
        default=False, description="Whether the tool requires gradients"
    )

    def __init__(self, require_grad: bool = False, **kwargs):
        """初始化实例。"""
        super().__init__(require_grad=require_grad, **kwargs)

    async def __call__(self, url: str, **kwargs) -> ToolResponse:
        """执行组件调用并返回结果。"""
        try:
            res = await fetch_url(url)
            if not res:
                logger.error(f"Failed to fetch content from {url}")
                return ToolResponse(
                    success=False,
                    message=f"Failed to fetch content from {url}",
                    extra=ToolExtra(data={"url": url, "status": "failed"}),
                )
            formatted = f"Title: {res.title}\nContent: {res.markdown}"
            return ToolResponse(
                success=True,
                message=formatted,
                extra=ToolExtra(
                    data={
                        "url": url,
                        "status": "success",
                        "content_length": len(formatted),
                        "title": res.title,
                        "markdown_length": len(res.markdown) if res.markdown else 0,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error fetching content: {e}")
            return ToolResponse(
                success=False,
                message=f"Failed to fetch content: {e}",
                extra=ToolExtra(
                    data={
                        "url": url,
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                ),
            )
