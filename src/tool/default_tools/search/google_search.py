from typing import List, Optional, Dict, Any
import json
from dotenv import load_dotenv
load_dotenv(verbose=True)

import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import unquote
from time import sleep
from pydantic import Field

from src.tool.default_tools.search.types import SearchItem
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.logger import logger
from src.registry import TOOL
from googlesearch.user_agents import get_useragent

def _req(term, results, tbs, lang, start, proxies, timeout, safe, ssl_verify, region):

    params = {
        "q": term,
        "num": results + 2,  # 处理输入参数。
        "hl": lang,
        "start": start,
        "safe": safe,
        "gl": region,
    }
    if tbs is not None:
        params["tbs"] = tbs

    resp = requests.get(
        url="https://www.google.com/search",
        headers={
            "User-Agent": get_useragent(),
            "Accept": "*/*"
        },
        params=params,
        proxies=proxies,
        timeout=timeout,
        verify=ssl_verify,
        cookies = {
            'CONSENT': 'PENDING+987', # 说明相关实现细节。
            'SOCS': 'CAESHAgBEhIaAB',
        }
    )
    resp.raise_for_status()
    return resp


def google_search(term,
                  num_results=10,
                  tbs=None,
                  lang="en",
                  proxy=None,
                  advanced=False,
                  sleep_interval=0,
                  timeout=5,
                  safe="active",
                  ssl_verify=None,
                  region=None,
                  start_num=0,
                  unique=False):
    """实现 `google_search` 的业务逻辑。"""

    # 初始化相关状态。
    proxies = {"https": proxy, "http": proxy} if proxy and (proxy.startswith("https") or proxy.startswith("http")) else None

    start = start_num
    fetched_results = 0  # 组装并返回结果。
    fetched_links = set() # 加载所需数据。

    while fetched_results < num_results:
        # 处理输入参数。
        resp = _req(term,
                    num_results - start,
                    tbs,
                    lang,
                    start,
                    proxies,
                    timeout,
                    safe,
                    ssl_verify,
                    region)

        # 处理文件与路径。
        # 说明相关实现细节。
        # 持久化相关数据。

        # 转换并规范化数据。
        soup = BeautifulSoup(resp.text, "html.parser")
        result_block = soup.find_all("div", class_="ezO2md")
        new_results = 0  # 组装并返回结果。

        for result in result_block:
            # 组装并返回结果。
            link_tag = result.find("a", href=True)
            # 说明相关实现细节。
            title_tag = link_tag.find("span", class_="CVA68e") if link_tag else None
            # 组装并返回结果。
            description_tag = result.find("span", class_="FrIlee")

            # 校验输入与当前状态。
            if link_tag and title_tag and description_tag:
                # 说明相关实现细节。
                link = unquote(link_tag["href"].split("&")[0].replace("/url?q=", "")) if link_tag else ""
            # 说明相关实现细节。
            link = unquote(link_tag["href"].split("&")[0].replace("/url?q=", "")) if link_tag else ""
            # 加载所需数据。
            if link in fetched_links and unique:
                continue  # 组装并返回结果。
            # 更新相关状态。
            fetched_links.add(link)
            # 说明相关实现细节。
            title = title_tag.text if title_tag else ""
            # 说明相关实现细节。
            description = description_tag.text if description_tag else ""
            # 组装并返回结果。
            fetched_results += 1
            # 组装并返回结果。
            new_results += 1
            # 组装并返回结果。
            if advanced:
                yield SearchItem(
                    title=title,
                    url=link,
                    date=None,
                    position=None,
                    source=None,
                    description=description,
                )
            else:
                yield link  # 说明相关实现细节。

            if fetched_results >= num_results:
                break  # 组装并返回结果。

        if new_results == 0:
            # 说明相关实现细节。
            # 组装并返回结果。
            break  # 组装并返回结果。

        start += 10  # 更新相关状态。
        sleep(sleep_interval)

def search(params):
    """实现 `search` 的业务逻辑。"""

    base_url = os.getenv("SKYWORK_GOOGLE_SEARCH_API", None)

    query = params.get("q", "")
    filter_year = params.get("filter_year", None)

    # 检索所需信息。
    if base_url is not None:
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            items = response.json()
        else:
            raise ValueError(response.json())

        if "organic" not in items.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")

        results = []
        if "organic" in items:
            for idx, page in enumerate(items["organic"]):
                title = page.get("title", f"Google Result {idx + 1}")
                url = page.get("link", "")
                position = page.get("position", idx + 1)
                description = page.get("snippet", None)
                date = page.get("date", None)
                source = page.get("source", None)

                results.append(
                    SearchItem(
                        title=title,
                        url=url,
                        date=date,
                        position=position,
                        source=source,
                        description=description,
                    )
                )
        return results

    else: # 检索所需信息。
        response = google_search(
            term=params["q"],
            num_results=params["num"],
            tbs=params.get("tbs", None),
            lang="en",
            proxy=None,
            advanced=True,
            sleep_interval=0,
            timeout=5,
        )

        results = []
        for item in response:
            results.append(item)

        return results

@TOOL.register_module(force=True)
class GoogleSearch(Tool):
    """定义 `GoogleSearch`，封装相关数据与行为。"""

    name: str = "google_search"
    description: str = (
        "a search engine using Google. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")

    async def _perform_search(
        self,
        query: str,
        num_results: int = 10,
        filter_year: Optional[int] = None,
        *args, **kwargs
    ) -> List[SearchItem]:
        """实现 `_perform_search` 的业务逻辑。"""
        params = {
            "q": query,
            "num": num_results,
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        results = search(params)

        return results

    async def __call__(
        self,
        query: str,
        num_results: Optional[int] = 10,
        country: Optional[str] = "us",
        lang: Optional[str] = "en",
        filter_year: Optional[int] = None,
        **kwargs
    ) -> ToolResponse:
        """执行组件调用并返回结果。"""
        try:
            # 检索所需信息。
            search_items = await self._perform_search(
                query,
                num_results=num_results,
                filter_year=filter_year
            )

            # 组装并返回结果。
            results_json = json.dumps([{
                "title": item.title,
                "url": item.url,
                "description": item.description or ""
            } for item in search_items], ensure_ascii=False, indent=4)

            message = f"Google search results for query: {query}\n\n{results_json}"

            return ToolResponse(
                success=True,
                message=message,
                extra=ToolExtra(
                    data={
                        "query": query,
                        "num_results": len(search_items),
                        "search_items": search_items,
                        "engine": "google",
                        "filter_year": filter_year
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error in Google search: {e}")
            return ToolResponse(
                success=False,
                message=f"Error in Google search: {str(e)}"
            )
