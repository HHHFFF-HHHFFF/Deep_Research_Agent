"""提供file reader相关实现。"""

import os
from typing import Optional, Dict, Any
from pydantic import Field

from src.tool.types import Tool, ToolResponse, ToolExtra
from src.registry import TOOL
from src.logger import logger

_FILE_READER_DESCRIPTION = """File reader tool for reading file contents.

🎯 BEST FOR: Reading text files with optional line range:
- Read entire file content
- Read specific line range (start_line to end_line)
- Useful for reviewing reports, logs, code files, etc.

📋 Parameters:
- file_path: Path to the file to read (required)
- start_line: Starting line number (optional, 1-indexed)
- end_line: Ending line number (optional, inclusive)

💡 Examples:
- Read entire file: {"name": "read", "args": {"file_path": "/path/to/report.md"}}
- Read lines 1-400: {"name": "read", "args": {"file_path": "/path/to/report.md", "start_line": 1, "end_line": 400}}
- Read from line 400 to end: {"name": "read", "args": {"file_path": "/path/to/report.md", "start_line": 400}}
"""


@TOOL.register_module(force=True)
class FileReaderTool(Tool):
    """定义 `FileReaderTool`，封装相关数据与行为。"""

    name: str = "read"
    description: str = _FILE_READER_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")

    def __init__(self, require_grad: bool = False, **kwargs):
        """初始化实例。"""
        super().__init__(require_grad=require_grad, **kwargs)

    async def __call__(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs
    ) -> ToolResponse:
        """执行组件调用并返回结果。"""
        try:
            # 校验输入与当前状态。
            if not file_path or not file_path.strip():
                return ToolResponse(
                    success=False,
                    message="Error: file_path is required."
                )

            file_path = file_path.strip()

            # 校验输入与当前状态。
            if not os.path.exists(file_path):
                return ToolResponse(
                    success=False,
                    message=f"Error: File not found: {file_path}"
                )

            # 校验输入与当前状态。
            if not os.path.isfile(file_path):
                return ToolResponse(
                    success=False,
                    message=f"Error: Path is not a file: {file_path}"
                )

            # 加载所需数据。
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            total_lines = len(lines)

            # 说明相关实现细节。
            if start_line is not None or end_line is not None:
                # 转换并规范化数据。
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else total_lines

                # 说明相关实现细节。
                if start_idx < 0:
                    start_idx = 0
                if start_idx > total_lines:
                    start_idx = total_lines  # 组装并返回结果。
                if end_idx > total_lines:
                    end_idx = total_lines  # 处理文件与路径。
                if end_idx < 0:
                    end_idx = 0

                # 说明相关实现细节。
                if start_idx >= end_idx:
                    if start_line and start_line > total_lines:
                        return ToolResponse(
                            success=True,
                            message=f"File: {file_path}\nNote: start_line ({start_line}) exceeds file length ({total_lines} lines). No content to display.",
                            extra=ToolExtra(
                                file_path=file_path,
                                data={
                                    "content": "",
                                    "total_lines": total_lines
                                }
                            )
                        )
                    # 说明相关实现细节。
                    return ToolResponse(
                        success=True,
                        message=f"File: {file_path}\nNote: Line range {start_line}-{end_line} is empty or invalid. File has {total_lines} lines.",
                        extra=ToolExtra(
                            file_path=file_path,
                            data={
                                "content": "",
                                "total_lines": total_lines
                            }
                        )
                    )

                # 说明相关实现细节。
                selected_lines = lines[start_idx:end_idx]
                content = ''.join(selected_lines)

                # 说明相关实现细节。
                numbered_content = ""
                for i, line in enumerate(selected_lines, start=start_idx + 1):
                    numbered_content += f"{i:6}|{line}"

                # 说明相关实现细节。
                adjusted_note = ""
                if end_line and end_line > total_lines:
                    adjusted_note = f" (requested end_line {end_line} adjusted to {total_lines})"

                logger.info(f"| 📖 Read file {file_path} lines {start_idx + 1}-{end_idx}")
                return ToolResponse(
                    success=True,
                    message=f"File: {file_path}\nLines: {start_idx + 1}-{end_idx} (of {total_lines} total){adjusted_note}\n\n{numbered_content}",
                    extra=ToolExtra(
                        file_path=file_path,
                        data={
                            "content": content,
                            "start_line": start_idx + 1,
                            "end_line": end_idx,
                            "total_lines": total_lines
                        }
                    )
                )
            else:
                # 组装并返回结果。
                content = ''.join(lines)

                # 说明相关实现细节。
                numbered_content = ""
                for i, line in enumerate(lines, start=1):
                    numbered_content += f"{i:6}|{line}"

                message = f"File: {file_path}\nTotal lines: {total_lines}\n\n{numbered_content}"

                logger.info(f"| 📖 Read file {file_path} ({total_lines} lines)")
                return ToolResponse(
                    success=True,
                    message=message,
                    extra=ToolExtra(
                        file_path=file_path,
                        data={
                            "content": content,
                            "total_lines": total_lines
                        }
                    )
                )

        except UnicodeDecodeError:
            return ToolResponse(
                success=False,
                message=f"Error: Cannot read file as text (binary file?): {file_path}"
            )
        except Exception as e:
            logger.error(f"| ❌ Error reading file: {e}")
            return ToolResponse(
                success=False,
                message=f"Error reading file: {str(e)}"
            )
