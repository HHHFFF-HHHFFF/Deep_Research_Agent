"""提供reporter相关实现。"""

import asyncio
import json
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.message import HumanMessage, SystemMessage
from src.model import model_manager
from src.registry import TOOL
from src.tool.types import Tool, ToolExtra, ToolResponse
from src.utils import (
    assemble_project_path,
    dedent,
    file_lock,
    read_text_file,
    write_text_file,
)


class ContentItem(BaseModel):
    """定义 `ContentItem`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    content: str = Field(description="The content of the item")
    summary: str = Field(description="The summary of the item")
    reference_ids: list[int] = Field(description="The reference IDs of the item")


class ReferenceItem(BaseModel):
    """定义 `ReferenceItem`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: int = Field(description="The ID of the reference")
    description: str = Field(description="The brief description of the reference")
    url: str | None = Field(default=None, description="The URL of the reference")


class ReportItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    content: ContentItem = Field(description="The content of the item")
    references: list[ReferenceItem] = Field(description="The references of the item")


class Report(BaseModel):
    """定义 `Report`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    title: str = Field(description="The title of the report")
    items: list[ReportItem] = Field(default=[], description="The items of the report")
    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="The model to use for extraction",
    )
    report_file_path: str | None = Field(
        default=None, description="The file path where the report will be saved"
    )

    def __init__(
        self,
        model_name: str | None = None,
        report_file_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_name is not None:
            self.model_name = model_name
        if report_file_path is not None:
            self.report_file_path = report_file_path

    async def add_item(
        self, file_path: str | None = None, content: str | dict[str, Any] | None = None
    ):
        """添加与 `add_item` 对应的数据或状态。"""
        # 加载所需数据。
        file_content = ""
        if file_path and os.path.exists(file_path):
            try:
                file_content = await read_text_file(file_path)
            except Exception as e:
                # 加载所需数据。
                file_content = f"[Note: Failed to read file {file_path}: {e!s}]"

        # 处理输入参数。
        if isinstance(content, dict):
            # 转换并规范化数据。
            input_text = json.dumps(content, indent=4, ensure_ascii=False)
        else:
            input_text = str(content) if content else ""

        # 处理文件与路径。
        combined_content = input_text
        if file_content:
            if combined_content:
                combined_content = f"{combined_content}\n\n--- File Content from {file_path} ---\n\n{file_content}"
            else:
                combined_content = (
                    f"--- File Content from {file_path} ---\n\n{file_content}"
                )

        # 创建所需对象。
        prompt = dedent(f"""Extract and structure the following content into a report item with content, summary, and references.

        Input Content:
        ```json
        {combined_content}
        ```

        Please extract:
        1. **Content**: The main content text (preserve the original content exactly, including all citations in markdown link format [1](url), [2](url), [3](url), etc.)
        2. **Summary**: A concise 2-3 sentence summary of the content
        3. **Reference IDs**: List of integer IDs that reference sources mentioned in the content (e.g., if content has [1](url), [2](url), extract [1, 2])
        4. **References**: List of reference items, each with:
           - id: Integer ID matching the reference IDs found in the content
           - description: Brief description of the reference source (e.g., file path, URL, document name)
           - url: URL for the reference (REQUIRED - extract from content, description, or file_path)

        IMPORTANT REQUIREMENTS:
        - **Citation Format**: Citations MUST be in markdown link format: [1](url), [2](url), [3](url), etc. (NOT just [1], [2], [3])
        - **Preserve Citations**: The content field MUST include all citation markers in [number](url) format exactly as they appear in the input
        - **Extract Reference IDs**: Parse all citation numbers from the content (e.g., [1](url), [2](url)
        - **Match References**: Each reference_id in the content must have a corresponding ReferenceItem with matching id
        - **Extract URLs**: Extract URLs from citations (e.g., [1](https://example.com) -> extract "https://example.com") and include them in both:
          - The citation format in content: [1](url)
          - The url field of the corresponding ReferenceItem
        - If the content contains citations like [1](url), [2](url), extract those numbers as reference_ids and create corresponding ReferenceItem entries with matching URLs
        - If citations are in [1] format without URLs, convert them to [1](url) format using the URL from the corresponding reference
        - If no citations are present, you may infer references from the content or use empty lists
        - If file_path is provided, include it in the references with an appropriate description and URL (convert file path to file:// URL format)

        Please Only Return the ReportItem object, no other text or explanation.
        """)

        messages = [
            SystemMessage(
                content="You are an expert at extracting structured information from content. Extract content, summaries, and references accurately."
            ),
            HumanMessage(content=prompt),
        ]

        # 组装并返回结果。
        response = await model_manager(
            model=self.model_name, messages=messages, response_format=ReportItem
        )

        if not response.success or not getattr(response, "extra", None):
            raise ValueError(f"Failed to extract report item: {response.message}")

        # 转换并规范化数据。
        report_item = response.extra.parsed_model

        # 说明相关实现细节。
        self.items.append(report_item)

        return report_item

    async def complete(self):
        """实现 `complete` 的业务逻辑。"""
        if not self.items:
            raise ValueError("Cannot complete report: no items found")

        if not self.report_file_path:
            raise ValueError("Cannot complete report: report_file_path is not set")

        # 说明相关实现细节。
        # 说明相关实现细节。
        all_references_dict: dict[str, ReferenceItem] = {}  # 说明相关实现细节。
        reference_key_to_id: dict[str, int] = {}  # 说明相关实现细节。

        def normalize_reference_key(ref: ReferenceItem) -> str:
            """实现 `normalize_reference_key` 的业务逻辑。"""
            # 说明相关实现细节。
            desc = ref.description.strip().lower() if ref.description else ""

            # 说明相关实现细节。
            url = ref.url.strip().lower() if ref.url else ""

            # 说明相关实现细节。
            if url:
                # 移除相关数据或组件。
                url_normalized = url.rstrip("/")
                return f"url:{url_normalized}"

            # 说明相关实现细节。
            if desc.startswith(("http://", "https://", "file://")):
                desc_normalized = desc.rstrip("/")
                return f"url:{desc_normalized}"

            # 说明相关实现细节。
            return f"desc:{desc}"

        for item in self.items:
            for ref in item.references:
                normalized_key = normalize_reference_key(ref)

                # 转换并规范化数据。
                if normalized_key in all_references_dict:
                    existing_ref = all_references_dict[normalized_key]
                    # 说明相关实现细节。
                    if ref.url and not existing_ref.url:
                        existing_ref.url = ref.url
                    # 说明相关实现细节。
                    if ref.description and len(ref.description) > len(
                        existing_ref.description
                    ):
                        existing_ref.description = ref.description
                else:
                    # 说明相关实现细节。
                    all_references_dict[normalized_key] = ref
                    reference_key_to_id[normalized_key] = ref.id

        # 创建所需对象。
        unique_references = list(all_references_dict.values())
        reference_mapping: dict[int, int] = {}  # 说明相关实现细节。

        # 创建所需对象。
        for new_id, (normalized_key, ref) in enumerate(
            all_references_dict.items(), start=1
        ):
            # 说明相关实现细节。
            for item in self.items:
                for old_ref in item.references:
                    old_normalized_key = normalize_reference_key(old_ref)
                    if old_normalized_key == normalized_key:
                        reference_mapping[old_ref.id] = new_id

        # 创建所需对象。
        reference_urls: dict[int, str] = {}  # 说明相关实现细节。
        for new_id, ref in enumerate(unique_references, start=1):
            description = ref.description
            # 说明相关实现细节。
            # 加载所需数据。
            if description.startswith(("http://", "https://")):
                reference_urls[new_id] = description
            # 转换并规范化数据。
            elif (
                os.path.exists(description) or "/" in description or "\\" in description
            ):
                # 转换并规范化数据。
                abs_path = (
                    os.path.abspath(description)
                    if not os.path.isabs(description)
                    else description
                )
                reference_urls[new_id] = f"file://{abs_path}"
            # 说明相关实现细节。
            else:
                # 说明相关实现细节。
                url_match = re.search(r"(https?://[^\s]+)", description)
                if url_match:
                    reference_urls[new_id] = url_match.group(1)
                else:
                    # 执行回退或重试逻辑。
                    reference_urls[new_id] = description

        # 更新相关状态。
        updated_contents = []
        for item in self.items:
            content = item.content.content
            reference_ids = item.content.reference_ids

            # 更新相关状态。
            def replace_citation(match):
                old_id_str = match.group(1)
                try:
                    old_id = int(old_id_str)
                    new_id = reference_mapping.get(old_id)
                    if new_id is not None:
                        url = reference_urls.get(new_id, f"#ref{new_id}")
                        return f"[{new_id}]({url})"
                    return match.group(0)  # 说明相关实现细节。
                except ValueError:
                    return match.group(0)  # 说明相关实现细节。

            # 说明相关实现细节。
            # 转换并规范化数据。
            updated_content = re.sub(
                r"\[(\d+)\]?(?:\([^)]+\))?", replace_citation, content
            )

            # 更新相关状态。
            updated_reference_ids = [
                reference_mapping.get(rid, rid) for rid in reference_ids
            ]
            # 移除相关数据或组件。
            updated_reference_ids = sorted(set(updated_reference_ids))

            updated_contents.append(
                {
                    "content": updated_content,
                    "summary": item.content.summary,
                    "reference_ids": updated_reference_ids,
                }
            )

        # 创建所需对象。
        renumbered_references = []
        for new_id, ref in enumerate(unique_references, start=1):
            # 说明相关实现细节。
            url = ref.url if ref.url else reference_urls.get(new_id, ref.description)
            renumbered_references.append(
                {"id": new_id, "description": ref.description, "url": url}
            )

        # 创建所需对象。
        items_text = "\n\n".join(
            [
                f"## Item {i + 1}\n\n**Summary:** {item['summary']}\n\n**Content:**\n{item['content']}\n\n**Reference IDs:** {item['reference_ids']}"
                for i, item in enumerate(updated_contents)
            ]
        )

        # 说明相关实现细节。
        references_text = "\n".join(
            [
                f"[{ref['id']}]({ref['url']}) {ref['description']}"
                for ref in renumbered_references
            ]
        )
        # 说明相关实现细节。
        if references_text:
            references_text = "\n" + references_text.replace("\n", "\n\n") + "\n"
        # 说明相关实现细节。
        if references_text:
            references_text = "\n" + references_text.replace("\n", "\n\n") + "\n"

        prompt = dedent(f"""Generate a complete, well-structured markdown report based on the following report items and references.

        Report Title: {self.title}

        Report Items:
        {items_text}

        References:
        {references_text}

        Please generate a comprehensive markdown report that:
        1. **Starts with the title** as a main heading (# {self.title})
        2. **Organizes content logically** - Group related items into sections with appropriate headings
        3. **Preserves all citations** - Keep all citation markers [1](url), [2](url), [3](url), etc. exactly as they appear in the content (with URLs)
        4. **Integrates summaries** - Use item summaries to create smooth transitions and context
        5. **Maintains coherence** - Ensure the report flows logically from introduction to conclusion
        6. **Includes References section** - Add a "## References" section at the end listing all references in numerical order with URLs, each on a separate line with proper spacing:
           ```
           ## References

           [1](url1) Reference description 1

           [2](url2) Reference description 2

           ...
           ```

        IMPORTANT REQUIREMENTS:
        - **Preserve All Citations**: Keep all citation markers [1](url), [2](url), [3](url) exactly as they appear in the content (with URLs)
        - **Citation Format**: All citations MUST be in markdown link format: [number](url)
        - **Preserve All Facts**: Do not modify facts, numbers, data, or specific details from the content
        - **Use All Content**: Include all content from all items, organized logically
        - **Complete References**: Include all references in the References section, numbered sequentially [1](url), [2](url), [3](url), etc.
        - **Markdown Format**: Use proper markdown formatting (headings, lists, paragraphs, etc.)
        - **Professional Style**: Write in a professional, academic report style

        ⚠️ CRITICAL FILE PATH REQUIREMENTS:
        - **MUST use absolute paths** for all file references in markdown content (images, links, file paths, etc.)
        - When referencing images or files in the report content, use absolute paths like:
          - ✅ Correct: `![Chart](/path/to/workdir/esg_agent/tool/plotter/chart.png)`
          - ✅ Correct: `[Link](/path/to/workdir/esg_agent/tool/data/file.pdf)`
          - ❌ Wrong: `![Chart](chart.png)` or `![Chart](./chart.png)` or `![Chart](../chart.png)`
        - Absolute paths ensure proper rendering in markdown viewers and editors
        - If any file paths appear in the content or references, they MUST be absolute paths

        Return ONLY the complete markdown report content, no explanations or additional text.
        """)

        messages = [
            SystemMessage(
                content="You are an expert report writer specializing in creating comprehensive, well-structured reports with proper citations and references."
            ),
            HumanMessage(content=prompt),
        ]

        # 处理模型调用。
        response = await model_manager(model=self.model_name, messages=messages)

        if not response.success:
            raise ValueError(f"Failed to generate report: {response.message}")

        report_content = response.message.strip()

        # 转换并规范化数据。
        if "## References" not in report_content and "References" not in report_content:
            # 说明相关实现细节。
            report_content += f"\n\n## References\n\n{references_text}\n"
        else:
            # 校验输入与当前状态。
            # 转换并规范化数据。
            report_content = re.sub(
                r"## References.*?(?=\n##|\Z)",
                f"## References\n\n{references_text}\n",
                report_content,
                flags=re.DOTALL,
            )

        # 说明相关实现细节。
        def add_url_to_citation(match):
            citation_num = match.group(1)
            # 加载所需数据。
            if match.group(0).count("(") == 0:  # 说明相关实现细节。
                # 说明相关实现细节。
                citation_id = int(citation_num)
                url = reference_urls.get(citation_id, f"#ref{citation_id}")
                return f"[{citation_num}]({url})"
            return match.group(0)  # 加载所需数据。

        # 加载所需数据。
        report_content = re.sub(r"\[(\d+)\](?!\()", add_url_to_citation, report_content)

        # 持久化相关数据。
        os.makedirs(os.path.dirname(self.report_file_path), exist_ok=True)
        async with file_lock(self.report_file_path):
            await write_text_file(self.report_file_path, report_content)

        return report_content


_REPORT_DESCRIPTION = """Report tool for managing and refining markdown reports.

🎯 BEST FOR: Creating, editing, and refining analysis reports.

📋 Actions:
- add: Add new content to the report
  - args:
    - file_path (Optional[str]) - Path to a markdown file to add (file content will be read and added)
    - content (Optional[Union[str, Dict[str, Any]]]) - The content to add as string or dictionary
    - At least one of content or file_path must be provided
  - Automatically generates summary and updates content list
  - Appends content to report.md

- complete: Complete and optimize the entire report
  - Reads all summaries and optimizes content for coherence and logic
  - Updates report.md with optimized content

💡 Workflow:
1. Use `add` multiple times to incrementally add content (from strings or dictionaries or files)
2. Use `complete` to optimize the entire report with LLM

Example: {"name": "reporter", "args": {"action": "add", "file_path": "/path/to/file.md", "content": "The content of the file."}}
Example: {"name": "reporter", "args": {"action": "complete"}}
"""


@TOOL.register_module(force=True)
class ReporterTool(Tool):
    """定义 `ReporterTool`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "reporter"
    description: str = _REPORT_DESCRIPTION
    metadata: dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(
        default=False, description="Whether the tool requires gradients"
    )

    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="The model to use for code generation.",
    )

    # 配置相关参数。
    base_dir: str = Field(
        default="workdir/reporter", description="The base directory for saving reports."
    )

    def __init__(
        self,
        base_dir: str | None = None,
        model_name: str | None = None,
        require_grad: bool = False,
        **kwargs,
    ):
        """初始化实例。"""
        super().__init__(require_grad=require_grad, **kwargs)

        if model_name is not None:
            self.model_name = model_name

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(self.base_dir)

        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)

        # 处理记忆或缓存状态。
        # 说明相关实现细节。
        self._report_cache: dict[str, Report] = {}
        # 执行异步任务。
        self._report_locks: dict[str, asyncio.Lock] = {}
        # 处理记忆或缓存状态。
        self._cache_lock = asyncio.Lock()

    def _get_report_file_path(self, id: str) -> str | None:
        """实现 `_get_report_file_path` 的业务逻辑。"""
        if not self.base_dir:
            return None
        # 创建所需对象。
        safe_id = re.sub(r"[^\w\s-]", "", id).strip().replace(" ", "_")
        if not safe_id:
            safe_id = "report"
        md_filename = f"{safe_id}.md"
        return os.path.join(self.base_dir, md_filename)

    async def _get_or_create_report(self, id: str) -> tuple[Report, asyncio.Lock]:
        """实现 `_get_or_create_report` 的业务逻辑。"""
        async with self._cache_lock:
            # 创建所需对象。
            if id not in self._report_locks:
                self._report_locks[id] = asyncio.Lock()

            # 创建所需对象。
            if id not in self._report_cache:
                # 处理文件与路径。
                report_file_path = self._get_report_file_path(id)
                # 说明相关实现细节。
                self._report_cache[id] = Report(
                    title=id,
                    model_name=self.model_name,
                    report_file_path=report_file_path,
                )
                logger.info(
                    f"| 📝 Created new report cache for id: {id} (file_path: {report_file_path})"
                )
            else:
                logger.info(
                    f"| 📂 Using existing report cache for id: {id} (items: {len(self._report_cache[id].items)})"
                )

            return self._report_cache[id], self._report_locks[id]

    async def _cleanup_report(self, id: str):
        """实现 `_cleanup_report` 的业务逻辑。"""
        async with self._cache_lock:
            if id in self._report_cache:
                del self._report_cache[id]
                logger.info(f"| 🧹 Removed report from cache: {id}")
            if id in self._report_locks:
                del self._report_locks[id]

    async def __call__(
        self,
        action: str,
        file_path: str | None = None,
        content: str | dict[str, Any] | None = None,
        **kwargs,
    ) -> ToolResponse:
        """执行组件调用并返回结果。"""
        try:
            # 处理工具调用。
            ctx = kwargs.get("ctx")
            id = ctx.id

            # 创建所需对象。
            # 更新相关状态。
            report, report_lock = await self._get_or_create_report(id)

            # 加载所需数据。
            async with report_lock:
                logger.info(
                    f"| 📝 ReporterTool action: {action} (id: {id}, report_file_path: {report.report_file_path}, items: {len(report.items)})"
                )

                if action == "add":
                    if not content and not file_path:
                        return ToolResponse(
                            success=False,
                            message="At least one of 'content' or 'file_path' is required for add action.",
                        )
                    return await self._add_content(
                        report=report, file_path=file_path, content=content
                    )

                elif action == "complete":
                    result = await self._complete_report(report=report)
                    # 处理记忆或缓存状态。
                    await self._cleanup_report(id)
                    return result

                else:
                    return ToolResponse(
                        success=False,
                        message=f"Unknown action: {action}. Valid actions: add, complete",
                    )

        except Exception as e:
            logger.error(f"| ❌ Error in ReporterTool: {e}")
            import traceback

            return ToolResponse(
                success=False,
                message=f"Error in report action '{action}': {e!s}\n{traceback.format_exc()}",
            )

    async def _add_content(
        self,
        report: Report,
        file_path: str | None = None,
        content: str | dict[str, Any] | None = None,
    ) -> ToolResponse:
        """实现 `_add_content` 的业务逻辑。"""
        try:
            # 处理文件与路径。
            resolved_file_path = None
            is_markdown_file = False

            if file_path:
                if os.path.isabs(file_path):
                    resolved_file_path = file_path
                else:
                    # 处理文件与路径。
                    if self.base_dir:
                        potential_path = os.path.join(self.base_dir, file_path)
                        if os.path.exists(potential_path):
                            resolved_file_path = os.path.abspath(potential_path)
                        else:
                            resolved_file_path = os.path.abspath(file_path)
                    else:
                        resolved_file_path = os.path.abspath(file_path)

                # 校验输入与当前状态。
                if resolved_file_path and os.path.exists(resolved_file_path):
                    file_ext = os.path.splitext(resolved_file_path)[1].lower()
                    is_markdown_file = file_ext in [".md", ".markdown"]

            # 加载所需数据。
            # 处理文件与路径。
            if is_markdown_file:
                # 加载所需数据。
                report_item = await report.add_item(
                    file_path=resolved_file_path, content=content
                )
            else:
                # 加载所需数据。
                # 创建所需对象。
                file_note = (
                    f"Attached file: {resolved_file_path}" if resolved_file_path else ""
                )
                combined_content = content if content else ""
                if file_note:
                    if combined_content:
                        combined_content = f"{combined_content}\n\n{file_note}"
                    else:
                        combined_content = file_note

                report_item = await report.add_item(
                    file_path=resolved_file_path, content=combined_content
                )

            item_id = len(report.items)
            logger.info(
                f"| ✅ Content added: ID={item_id}, Summary={report_item.content.summary[:100]}..."
            )

            # 创建所需对象。
            message_parts = [
                f"📝 Content added successfully!\n\nID: {item_id}\nSummary: {report_item.content.summary}"
            ]
            if resolved_file_path:
                message_parts.append(f"\nFile: {resolved_file_path}")

            return ToolResponse(
                success=True,
                message="\n".join(message_parts),
                extra=ToolExtra(
                    file_path=report.report_file_path,
                    data={
                        "id": item_id,
                        "summary": report_item.content.summary,
                        "reference_ids": report_item.content.reference_ids,
                        "references": [
                            {
                                "id": ref.id,
                                "description": ref.description,
                                "url": ref.url,
                            }
                            for ref in report_item.references
                        ],
                        "total_items": len(report.items),
                        "source_file_path": resolved_file_path
                        if resolved_file_path
                        else None,
                    },
                ),
            )

        except Exception as e:
            logger.error(f"| ❌ Error adding content: {e}")
            import traceback

            return ToolResponse(
                success=False,
                message=f"Error adding content: {e!s}\n{traceback.format_exc()}",
            )

    async def _complete_report(self, report: Report) -> ToolResponse:
        """实现 `_complete_report` 的业务逻辑。"""
        try:
            if not report or not report.items:
                return ToolResponse(
                    success=False,
                    message="Report is empty. Add content first using the 'add' action.",
                )

            if not report.report_file_path:
                return ToolResponse(
                    success=False,
                    message="Report file path is not set. Cannot complete report.",
                )

            logger.info(f"| 📊 Completing report with {len(report.items)} items...")

            # 说明相关实现细节。
            final_report_content = await report.complete()

            logger.info(
                f"| ✅ Report completion successful ({len(final_report_content)} chars)"
            )

            return ToolResponse(
                success=True,
                message=f"📝 Report completed successfully!\n\nPath: {report.report_file_path}\n\nThe entire report has been generated with properly numbered citations and references.",
                extra=ToolExtra(
                    file_path=report.report_file_path,
                    data={
                        "path": report.report_file_path,
                        "items_count": len(report.items),
                        "report_length": len(final_report_content),
                        "title": report.title,
                    },
                ),
            )

        except Exception as e:
            logger.error(f"| ❌ Error completing report: {e}")
            import traceback

            return ToolResponse(
                success=False,
                message=f"Error completing report: {e!s}\n{traceback.format_exc()}",
            )
