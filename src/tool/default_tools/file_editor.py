"""提供file editor相关实现。"""

import os
from typing import Any

from pydantic import Field

from src.logger import logger
from src.registry import TOOL
from src.tool.types import Tool, ToolExtra, ToolResponse
from src.utils import read_lines_file, write_lines_file

_FILE_EDITOR_DESCRIPTION = """File editor tool for editing file contents with multiple operations.

🎯 BEST FOR: Editing text files with line-based operations:
- Replace specific line ranges with new content
- Append content to the end of a file
- Perform multiple edits in a single call

📋 Parameters:
- file_path: Path to the file to edit (required)
- edits: List of edit operations (required), each operation is a dict with:
  - start_line: Starting line number (1-indexed, optional)
  - end_line: Ending line number (inclusive, optional)
  - content: New content to insert (required)

  If start_line and end_line are not provided, content is appended to the end.
  If only start_line is provided, content is inserted at that line.
  If both are provided, lines from start_line to end_line are replaced with content.

💡 Examples:
- Append to end:
  {"name": "edit", "args": {"file_path": "/path/to/report.md", "edits": [{"content": "## New Section\\n\\nContent here."}]}}

- Replace lines 10-15:
  {"name": "edit", "args": {"file_path": "/path/to/report.md", "edits": [{"start_line": 10, "end_line": 15, "content": "New content"}]}}

- Insert at line 5:
  {"name": "edit", "args": {"file_path": "/path/to/report.md", "edits": [{"start_line": 5, "content": "Inserted line\\n"}]}}

- Multiple edits (applied from bottom to top to preserve line numbers):
  {"name": "edit", "args": {"file_path": "/path/to/report.md", "edits": [
    {"start_line": 20, "end_line": 25, "content": "Replace section 2"},
    {"start_line": 5, "end_line": 10, "content": "Replace section 1"}
  ]}}
"""


@TOOL.register_module(force=True)
class FileEditorTool(Tool):
    """定义 `FileEditorTool`，封装相关数据与行为。"""

    name: str = "edit"
    description: str = _FILE_EDITOR_DESCRIPTION
    metadata: dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(
        default=False, description="Whether the tool requires gradients"
    )

    def __init__(self, require_grad: bool = False, **kwargs):
        """初始化实例。"""
        super().__init__(require_grad=require_grad, **kwargs)

    async def __call__(
        self, file_path: str, edits: list[dict[str, Any]], **kwargs
    ) -> ToolResponse:
        """执行组件调用并返回结果。"""
        try:
            # 校验输入与当前状态。
            if not file_path or not file_path.strip():
                return ToolResponse(
                    success=False, message="Error: file_path is required."
                )

            file_path = file_path.strip()

            # 校验输入与当前状态。
            if not os.path.exists(file_path):
                return ToolResponse(
                    success=False, message=f"Error: File not found: {file_path}"
                )

            # 校验输入与当前状态。
            if not os.path.isfile(file_path):
                return ToolResponse(
                    success=False, message=f"Error: Path is not a file: {file_path}"
                )

            # 校验输入与当前状态。
            if not edits or not isinstance(edits, list):
                return ToolResponse(
                    success=False,
                    message="Error: edits must be a non-empty list of edit operations.",
                )

            # 加载所需数据。
            lines = await read_lines_file(file_path)

            original_lines = len(lines)

            # 校验输入与当前状态。
            normalized_edits = []
            for i, edit in enumerate(edits):
                if not isinstance(edit, dict):
                    return ToolResponse(
                        success=False,
                        message=f"Error: Edit at index {i} must be a dict.",
                    )

                if "content" not in edit:
                    return ToolResponse(
                        success=False,
                        message=f"Error: Edit at index {i} must have 'content' field.",
                    )

                content = edit["content"]
                start_line = edit.get("start_line")
                end_line = edit.get("end_line")

                # 说明相关实现细节。
                if content and not content.endswith("\n"):
                    content += "\n"

                normalized_edits.append(
                    {
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": content,
                        "original_index": i,
                    }
                )

            # 说明相关实现细节。
            # 说明相关实现细节。
            def get_sort_key(edit):
                if edit["start_line"] is None:
                    return float("inf")  # 说明相关实现细节。
                return -edit["start_line"]  # 说明相关实现细节。

            sorted_edits = sorted(normalized_edits, key=get_sort_key)

            # 说明相关实现细节。
            edit_results = []
            for edit in sorted_edits:
                start_line = edit["start_line"]
                end_line = edit["end_line"]
                content = edit["content"]

                # 说明相关实现细节。
                content_lines = content.splitlines(keepends=True)
                if content and not content_lines:
                    content_lines = [content]

                if start_line is None and end_line is None:
                    # 说明相关实现细节。
                    lines.extend(content_lines)
                    edit_results.append(
                        {
                            "action": "append",
                            "lines_added": len(content_lines),
                            "at_line": len(lines) - len(content_lines) + 1,
                        }
                    )
                elif start_line is not None and end_line is None:
                    # 说明相关实现细节。
                    insert_idx = max(0, min(start_line - 1, len(lines)))
                    for j, line in enumerate(content_lines):
                        lines.insert(insert_idx + j, line)
                    edit_results.append(
                        {
                            "action": "insert",
                            "at_line": insert_idx + 1,
                            "lines_added": len(content_lines),
                        }
                    )
                else:
                    # 说明相关实现细节。
                    start_idx = max(0, start_line - 1)
                    end_idx = min(end_line, len(lines)) if end_line else start_idx + 1

                    # 说明相关实现细节。
                    start_idx = min(start_idx, len(lines))
                    end_idx = min(end_idx, len(lines))
                    end_idx = max(end_idx, start_idx)

                    lines_removed = end_idx - start_idx

                    # 移除相关数据或组件。
                    del lines[start_idx:end_idx]
                    for j, line in enumerate(content_lines):
                        lines.insert(start_idx + j, line)

                    edit_results.append(
                        {
                            "action": "replace",
                            "start_line": start_idx + 1,
                            "end_line": end_idx,
                            "lines_removed": lines_removed,
                            "lines_added": len(content_lines),
                        }
                    )

            # 持久化相关数据。
            await write_lines_file(file_path, lines)

            new_lines = len(lines)

            # 创建所需对象。
            result_msg = f"File edited: {file_path}\n"
            result_msg += (
                f"Original lines: {original_lines} → New lines: {new_lines}\n\n"
            )
            result_msg += "Edits applied:\n"
            for i, result in enumerate(edit_results, 1):
                if result["action"] == "append":
                    result_msg += f"  {i}. Appended {result['lines_added']} lines at line {result['at_line']}\n"
                elif result["action"] == "insert":
                    result_msg += f"  {i}. Inserted {result['lines_added']} lines at line {result['at_line']}\n"
                elif result["action"] == "replace":
                    result_msg += f"  {i}. Replaced lines {result['start_line']}-{result['end_line']} ({result['lines_removed']} lines) with {result['lines_added']} lines\n"

            message = result_msg

            logger.info(f"| ✏️ Edited file {file_path}: {len(edit_results)} operations")
            return ToolResponse(
                success=True,
                message=message,
                extra=ToolExtra(
                    file_path=file_path,
                    data={
                        "original_lines": original_lines,
                        "new_lines": new_lines,
                        "edits_applied": len(edit_results),
                        "edit_results": edit_results,
                    },
                ),
            )

        except UnicodeDecodeError:
            return ToolResponse(
                success=False,
                message=f"Error: Cannot edit file as text (binary file?): {file_path}",
            )
        except Exception as e:
            logger.error(f"| ❌ Error editing file: {e}")
            import traceback

            return ToolResponse(
                success=False,
                message=f"Error editing file: {e!s}\n{traceback.format_exc()}",
            )
