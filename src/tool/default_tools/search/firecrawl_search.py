from __future__ import annotations
from typing import Any, Optional, Dict, List, Type
import json
import os
from pydantic import ConfigDict, Field
from firecrawl import AsyncFirecrawlApp
from dotenv import load_dotenv
load_dotenv()

from src.tool.default_tools.search.types import SearchItem, SearchToolArgs
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.logger import logger
from src.registry import TOOL

@TOOL.register_module(force=True)
class FirecrawlSearch(Tool):
    """定义 `FirecrawlSearch`，封装相关数据与行为。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "firecrawl_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    api_key: Optional[str] = Field(default=None, description="Firecrawl API key")

    def __init__(self, **kwargs):
        """初始化实例。"""
        # 更新相关状态。
        super().__init__(**kwargs)
        self.api_key = os.getenv("FIRECRAWL_API_KEY")

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> FirecrawlSearch:
        """实现 `from_search_kwargs` 的业务逻辑。"""
        return cls(search_kwargs=search_kwargs, **kwargs)

    async def _search_firecrawl(self,
                                query: str,
                                num_results: int = 10,
                                filter_year: Optional[int] = 2025) -> List[SearchItem]:
        """实现 `_search_firecrawl` 的业务逻辑。"""
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable is required")

        results = []

        app = AsyncFirecrawlApp(api_key=self.api_key)
        search_kwargs = {
            "query": query,
            "limit": num_results,
        }

        # 说明相关实现细节。
        # 说明相关实现细节。
        if filter_year is None:
            filter_year = 2025  # 说明相关实现细节。

        if 1900 <= filter_year <= 2100:
            search_kwargs["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
        else:
            logger.warning(f"Invalid filter_year: {filter_year}. Expected 1900-2100. Ignoring date filter.")

        try:
            response = await app.search(**search_kwargs)
        except Exception as e:
            logger.error(f"Firecrawl API call failed: {e}")
            return results

        # 校验输入与当前状态。
        if response is None:
            logger.warning("Firecrawl search returned None response")
            return results

        # 组装并返回结果。
        logger.debug(f"Firecrawl response type: {type(response)}")
        logger.debug(f"Firecrawl response attributes: {dir(response) if hasattr(response, '__dict__') else 'N/A'}")

        # 校验输入与当前状态。
        web_results = None

        # 组装并返回结果。
        if hasattr(response, 'web') and response.web is not None:
            web_results = response.web
        elif hasattr(response, 'data') and response.data is not None:
            # 处理版本与历史记录。
            web_results = response.data
        elif hasattr(response, 'results') and response.results is not None:
            # 组装并返回结果。
            web_results = response.results
        elif isinstance(response, dict):
            # 组装并返回结果。
            web_results = response.get('web') or response.get('data') or response.get('results')
        elif isinstance(response, list):
            # 组装并返回结果。
            web_results = response
        else:
            # 组装并返回结果。
            try:
                if hasattr(response, 'model_dump'):
                    response_dict = response.model_dump()
                    web_results = response_dict.get('web') or response_dict.get('data') or response_dict.get('results')
                elif hasattr(response, 'dict'):
                    response_dict = response.dict()
                    web_results = response_dict.get('web') or response_dict.get('data') or response_dict.get('results')
            except Exception:
                pass

        if web_results is None:
            # 组装并返回结果。
            logger.warning(
                f"Firecrawl search response has no accessible results. "
                f"Response type: {type(response)}, Response: {str(response)[:200]}"
            )
            # 说明相关实现细节。
            if hasattr(response, '__dict__'):
                logger.debug(f"Response attributes: {list(response.__dict__.keys())}")
            elif hasattr(response, '__fields__'):
                logger.debug(f"Response fields: {list(response.__fields__.keys())}")
            return results

        # 组装并返回结果。
        try:
            for item in web_results:
                if item is None:
                    continue

                # 转换并规范化数据。
                if isinstance(item, dict):
                    title = item.get('title', '') or ""
                    url = item.get('url', '') or ""
                    description = item.get('description', '') or item.get('snippet', '') or ""
                else:
                    title = getattr(item, 'title', None) or ""
                    url = getattr(item, 'url', None) or ""
                    description = getattr(item, 'description', None) or getattr(item, 'snippet', None) or ""

                if url:  # 说明相关实现细节。
                    results.append(SearchItem(
                        title=title,
                        url=url,
                        description=description
                    ))
        except (TypeError, AttributeError) as e:
            logger.error(f"Error iterating over Firecrawl search results: {e}, web_results type: {type(web_results)}")
            return results

        return results

    async def __call__(
        self,
        query: str,
        num_results: Optional[int] = 5,
        country: Optional[str] = "us",
        lang: Optional[str] = "en",
        filter_year: Optional[int] = 2025,
        **kwargs
    ) -> ToolResponse:
        """执行组件调用并返回结果。"""

        try:

            # 检索所需信息。
            search_items = await self._search_firecrawl(query, num_results=num_results, filter_year=filter_year)

            # 组装并返回结果。
            results_json = json.dumps([{
                "title": item.title,
                "url": item.url,
                "description": item.description or ""
            } for item in search_items], ensure_ascii=False, indent=4)

            message = f"Firecrawl search results for query: {query}\n\n{results_json}"

            return ToolResponse(success=True, message=message, extra=ToolExtra(
                data={
                    "query": query,
                    "num_results": len(search_items),
                    "search_items": search_items,
                    "engine": "firecrawl"
                }
            ))

        except Exception as e:
            logger.error(f"Error in Firecrawl search: {e}")
            return ToolResponse(success=False, message=f"Error in Firecrawl search: {str(e)}")
