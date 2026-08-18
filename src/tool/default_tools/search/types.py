from pydantic import BaseModel, Field


class SearchItem(BaseModel):
    """定义 `SearchItem`，封装相关数据与行为。"""

    title: str = Field(description="The title of the search result")
    url: str = Field(description="The URL of the search result")
    date: str | None = Field(default=None, description="The date of the search result")
    position: int | None = Field(
        default=None, description="The position of the search result in the list"
    )
    source: str | None = Field(
        default=None, description="The source of the search result"
    )
    description: str | None = Field(
        default=None, description="A description or snippet of the search result"
    )

    def __str__(self) -> str:
        """返回便于阅读的文本表示。"""
        return f"{self.title} - {self.url} - {self.description or 'No description available'}"


class SearchToolArgs(BaseModel):
    """定义 `SearchToolArgs`，封装相关数据与行为。"""

    query: str = Field(description="The query to search for")
    num_results: int = Field(
        default=5, description="The number of search results to return, default is 5"
    )
    country: str | None = Field(
        default="us", description="The country to search in, default is us"
    )
    lang: str | None = Field(
        default="en", description="The language to search in, default is en"
    )
    filter_year: int | None = Field(
        default=None, description="The year to filter results by, default is None"
    )
