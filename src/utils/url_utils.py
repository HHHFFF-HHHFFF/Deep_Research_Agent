import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
load_dotenv(verbose=True)

from markitdown._base_converter import DocumentConverterResult
from crawl4ai import AsyncWebCrawler
from firecrawl import AsyncFirecrawlApp

# 检索所需信息。
DEFAULT_FETCH_TIMEOUT = 15  # 检索所需信息。

async def firecrawl_fetch_url(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT):
    """实现 `firecrawl_fetch_url` 的业务逻辑。"""
    try:
        app = AsyncFirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY", None))

        # 说明相关实现细节。
        response = await asyncio.wait_for(
            app.scrape(url),
            timeout=timeout
        )

        result = response.markdown
        return result
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        return None

async def fetch_crawl4ai_url(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT):
    """获取与 `fetch_crawl4ai_url` 对应的数据或状态。"""
    try:
        async with AsyncWebCrawler() as crawler:
            # 说明相关实现细节。
            response = await asyncio.wait_for(
                crawler.arun(url=url),
                timeout=timeout
            )

            if response:
                result = response.markdown
                return result
            else:
                return None
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        return None

async def fetch_url(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT) -> Optional[DocumentConverterResult]:
    """获取与 `fetch_url` 对应的数据或状态。"""
    try:
        # 说明相关实现细节。
        firecrawl_result = await firecrawl_fetch_url(url, timeout=timeout)

        if firecrawl_result:
            return DocumentConverterResult(
                markdown=firecrawl_result,
                title=f"Fetched content from {url}",
            )

        # 执行回退或重试逻辑。
        crawl4ai_result = await fetch_crawl4ai_url(url, timeout=timeout)
        if crawl4ai_result:
            return DocumentConverterResult(
                markdown=crawl4ai_result,
                title=f"Fetched content from {url}",
            )

    except Exception as e:
        return None

    return None

if __name__ == '__main__':
    import asyncio
    url = "https://www.google.com/"
    result = asyncio.run(firecrawl_fetch_url(url))
    print(result)
