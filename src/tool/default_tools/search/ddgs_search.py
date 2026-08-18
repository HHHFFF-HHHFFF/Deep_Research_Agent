import asyncio
import json
from typing import Any

from ddgs import DDGS
from pydantic import Field

from src.logger import logger
from src.registry import TOOL
from src.tool.default_tools.search.types import SearchItem
from src.tool.types import Tool, ToolExtra, ToolResponse


@TOOL.register_module(force=True)
class DDGSSearch(Tool):
    """定义 `DDGSSearch`，封装相关数据与行为。"""

    name: str = "ddgs_search"
    description: str = (
        "a search engine using DDGS. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    metadata: dict[str, Any] = Field(default={}, description="The metadata of the tool")

    async def _perform_search(
        self,
        query: str,
        num_results: int = 10,
        region: str | None = "wt-wt",
        safesearch: str | None = "moderate",
        *args,
        **kwargs,
    ) -> list[SearchItem]:
        """实现 `_perform_search` 的业务逻辑。"""

        # 加载所需数据。
        def _search_sync():
            results = []
            with DDGS() as ddgs:
                raw_results = ddgs.text(
                    query,
                    max_results=num_results,
                    region=region or "wt-wt",
                    safesearch=safesearch or "moderate",
                )
                for item in raw_results:
                    if isinstance(item, dict):
                        results.append(
                            SearchItem(
                                title=item.get(
                                    "title", f"DuckDuckGo Result {len(results) + 1}"
                                ),
                                url=item.get("href", ""),
                                description=item.get("body", None),
                            )
                        )
                    elif isinstance(item, str):
                        # 说明相关实现细节。
                        results.append(
                            SearchItem(
                                title=f"DuckDuckGo Result {len(results) + 1}",
                                url=item,
                                description=None,
                            )
                        )
            return results

        # 加载所需数据。
        return await asyncio.to_thread(_search_sync)

    async def __call__(
        self,
        query: str,
        num_results: int | None = 10,
        country: str | None = "us",
        lang: str | None = "en",
        filter_year: int | None = None,
        **kwargs,
    ) -> ToolResponse:
        """执行组件调用并返回结果。"""
        try:
            # 转换并规范化数据。
            # 说明相关实现细节。
            region = "wt-wt"  # 说明相关实现细节。
            if country and lang:
                # 转换并规范化数据。
                region_map = {
                    ("us", "en"): "us-en",
                    ("sg", "en"): "sg-en",
                    ("cn", "zh"): "cn-zh",
                    ("uk", "en"): "uk-en",
                }
                region = region_map.get(
                    (country.lower(), lang.lower()), f"{country.lower()}-{lang.lower()}"
                )
            elif country:
                region = f"{country.lower()}-en"
            elif lang:
                region = f"wt-{lang.lower()}"

            # 检索所需信息。
            search_items = await self._perform_search(
                query, num_results=num_results, region=region, safesearch="moderate"
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

            message = f"DuckDuckGo search results for query: {query}\n\n{results_json}"

            return ToolResponse(
                success=True,
                message=message,
                extra=ToolExtra(
                    data={
                        "query": query,
                        "num_results": len(search_items),
                        "search_items": search_items,
                        "engine": "duckduckgo",
                    }
                ),
            )

        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")
            return ToolResponse(
                success=False,
                message=f"Error in DuckDuckGo search: {e!s}",
            )
