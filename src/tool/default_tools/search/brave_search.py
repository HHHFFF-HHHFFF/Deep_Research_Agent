from __future__ import annotations

import json
import os
from typing import Any

import aiohttp
from dotenv import load_dotenv
from pydantic import ConfigDict, Field

load_dotenv()

from src.logger import logger
from src.registry import TOOL
from src.tool.default_tools.search.types import SearchItem
from src.tool.types import Tool, ToolExtra, ToolResponse


@TOOL.register_module(force=True)
class BraveSearch(Tool):
    """定义 `BraveSearch`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "brave_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    metadata: dict[str, Any] = Field(default={}, description="The metadata of the tool")
    api_key: str | None = Field(default=None, description="Brave Search API key")
    base_url: str = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, **kwargs):
        """初始化实例。"""
        # 更新相关状态。
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("BRAVE_SEARCH_API_KEY")

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> BraveSearch:
        """实现 `from_search_kwargs` 的业务逻辑。"""
        return cls(search_kwargs=search_kwargs, **kwargs)

    async def _search_brave(
        self,
        query: str,
        num_results: int = 10,
        country: str = "us",
        lang: str = "en",
        filter_year: int | None = 2025,
    ) -> list[SearchItem]:
        """实现 `_search_brave` 的业务逻辑。"""
        if not self.api_key:
            raise ValueError("BRAVE_SEARCH_API_KEY environment variable is required")

        results = []

        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }

        params = {
            "q": query,
            "count": num_results,
            "country": country.upper(),
            "search_lang": lang.lower(),
            "extra_snippets": "true",  # 转换并规范化数据。
        }

        # 说明相关实现细节。
        # 说明相关实现细节。
        if filter_year is None:
            filter_year = 2025  # 说明相关实现细节。

        if 1900 <= filter_year <= 2100:
            # 转换并规范化数据。
            # 处理输入参数。
            params["safesearch"] = "moderate"
            # 说明相关实现细节。
            # 说明相关实现细节。
        else:
            logger.warning(
                f"Invalid filter_year: {filter_year}. Expected 1900-2100. Ignoring date filter."
            )

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(self.base_url, headers=headers, params=params) as response,
            ):
                if not response.ok:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP error {response.status}: {error_text}")

                data = await response.json()
                web_results = data.get("web", {}).get("results", [])

                for item in web_results:
                    if item is None:
                        continue

                    title = item.get("title", "") or ""
                    url = item.get("url", "") or ""
                    description_parts = [
                        item.get("description", ""),
                        *item.get("extra_snippets", []),
                    ]
                    description = " ".join(filter(None, description_parts)) or ""

                    if url:  # 说明相关实现细节。
                        results.append(
                            SearchItem(title=title, url=url, description=description)
                        )
        except Exception as e:
            logger.error(f"Brave API call failed: {e}")
            return results

        return results

    async def __call__(
        self,
        query: str,
        num_results: int | None = 5,
        country: str | None = "us",
        lang: str | None = "en",
        filter_year: int | None = 2025,
        **kwargs,
    ) -> ToolResponse:
        """执行组件调用并返回结果。"""

        try:
            # 检索所需信息。
            search_items = await self._search_brave(
                query,
                num_results=num_results,
                country=country,
                lang=lang,
                filter_year=filter_year,
            )

            # 组装并返回结果。
            results_json = json.dumps(
                [
                    {
                        "title": item.title,
                        "url": item.url,
                        "description": item.description or "",
                    }
                    for item in search_items
                ],
                ensure_ascii=False,
                indent=4,
            )

            message = f"Brave search results for query: {query}\n\n{results_json}"

            return ToolResponse(
                success=True,
                message=message,
                extra=ToolExtra(
                    data={
                        "query": query,
                        "num_results": len(search_items),
                        "search_items": search_items,
                        "engine": "brave",
                    }
                ),
            )

        except Exception as e:
            logger.error(f"Error in Brave search: {e}")
            return ToolResponse(
                success=False,
                message=f"Error in Brave search: {e!s}",
            )
