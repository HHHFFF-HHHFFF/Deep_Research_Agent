"""提供file system environment相关实现。"""

from pathlib import Path
from typing import List, Optional, Any, Dict, Type, Union
from pydantic import BaseModel, Field, ConfigDict

from src.environment.filesystem.service import FileSystemService
from src.environment.filesystem.types import (
    FileReadRequest,
    FileWriteRequest,
    FileReplaceRequest,
    FileDeleteRequest,
    FileCopyRequest,
    FileMoveRequest,
    DirectoryCreateRequest,
    DirectoryDeleteRequest,
    FileListRequest,
    FileTreeRequest,
    FileSearchRequest,
    FileStatRequest,
    FileChangePermissionsRequest
)
from src.logger import logger
from src.utils import assemble_project_path
from src.environment.types import Environment
from src.utils import dedent
from src.environment.server import ecp
from src.registry import ENVIRONMENT

@ENVIRONMENT.register_module(force=True)
class FileSystemEnvironment(Environment):
    """定义 `FileSystemEnvironment`，封装相关数据与行为。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="file_system", description="The name of the file system environment.")
    description: str = Field(default="File system environment for file operations", description="The description of the file system environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the file system environment.",
        }
    }, description="The metadata of the file system environment.")
    require_grad: bool = Field(default=False, description="Whether the environment requires gradients")

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        max_file_size: int = 1024 * 1024,  # 1MB
        require_grad: bool = False,
        **kwargs
    ):
        """初始化实例。"""
        super().__init__(**kwargs)
        self.base_dir = assemble_project_path(base_dir)
        self.max_file_size = max_file_size

        # 初始化相关状态。
        self.file_system_service = FileSystemService(
            base_dir=self.base_dir
        )

    async def initialize(self) -> None:
        """初始化组件及其依赖资源。"""

        self.metadata["additional_rules"]["state"] = f"File System Environment at: {self.base_dir}. All `file_path` in the actions should be absolute paths based on this directory.\n"

        logger.info(f"| 🗂️ File System Environment initialized at: {self.base_dir}")

    async def cleanup(self) -> None:
        """释放组件占用的资源。"""
        logger.info("| 🧹 File System Environment cleanup completed")

    @ecp.action(name = "read",
                description = "Read a file from the file system.")
    async def read(self,
                    file_path: str,
                    start_line: Optional[int] = None,
                    end_line: Optional[int] = None,
                    **kwargs) -> Dict[str, Any]:
        """实现 `read` 的业务逻辑。"""
        try:
            request = FileReadRequest(
                path=Path(file_path),
                start_line=start_line,
                end_line=end_line
            )
            result = await self.file_system_service.read(request)

            extra = result.extra.copy() if result.extra else {}

            if result.success:
                if "content_text" in extra:
                    message = extra["content_text"]
                elif "content_bytes_length" in extra:
                    message = f"File read successfully. Content length: {extra['content_bytes_length']} bytes"
                else:
                    message = result.message
            else:
                message = result.message

            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to read file: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path}
            }

    @ecp.action(name = "write",
                description = "Write content to a file.")
    async def write(self,
                    file_path: str,
                    content: str,
                    mode: str = "w",
                    **kwargs) -> Dict[str, Any]:
        """实现 `write` 的业务逻辑。"""
        try:
            request = FileWriteRequest(
                path=Path(file_path),
                content=content,
                mode=mode
            )
            result = await self.file_system_service.write(request)

            extra = result.extra.copy() if result.extra else {}
            extra["file_path"] = file_path

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to write file: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path}
            }

    @ecp.action(name = "replace",
                description = "Replace a string in a file.")
    async def replace(self,
                       file_path: str,
                       old_string: str,
                       new_string: str,
                       start_line: Optional[int] = None,
                       end_line: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """实现 `replace` 的业务逻辑。"""
        try:
            request = FileReplaceRequest(
                path=Path(file_path),
                old_string=old_string,
                new_string=new_string,
                start_line=start_line,
                end_line=end_line
            )
            result = await self.file_system_service.replace(request)

            extra = result.extra.copy() if result.extra else {}
            extra["file_path"] = file_path

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to replace text: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path}
            }

    @ecp.action(name = "delete",
                description = "Delete a file from the file system.")
    async def delete(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """实现 `delete` 的业务逻辑。"""
        try:
            request = FileDeleteRequest(path=Path(file_path))
            result = await self.file_system_service.delete(request)

            extra = result.extra.copy() if result.extra else {}
            extra["file_path"] = file_path

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete file: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path}
            }

    @ecp.action(name = "copy",
                description = "Copy a file from source to destination.")
    async def copy(self, src_path: str, dst_path: str, **kwargs) -> Dict[str, Any]:
        """实现 `copy` 的业务逻辑。"""
        try:
            request = FileCopyRequest(src_path=Path(src_path), dst_path=Path(dst_path))
            result = await self.file_system_service.copy(request)

            extra = result.extra.copy() if result.extra else {}
            extra["src_path"] = src_path
            extra["dst_path"] = dst_path

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to copy file: {str(e)}",
                "extra": {"error": str(e), "src_path": src_path, "dst_path": dst_path}
            }

    @ecp.action(name = "move",
                description = "Move a file from source to destination.")
    async def move(self, src_path: str, dst_path: str, **kwargs) -> Dict[str, Any]:
        """实现 `move` 的业务逻辑。"""
        try:
            request = FileMoveRequest(
                src_path=Path(src_path),
                dst_path=Path(dst_path)
            )
            result = await self.file_system_service.rename(request)

            extra = result.extra.copy() if result.extra else {}
            extra["src_path"] = src_path
            extra["dst_path"] = dst_path

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to move file: {str(e)}",
                "extra": {"error": str(e), "src_path": src_path, "dst_path": dst_path}
            }

    @ecp.action(name = "rename",
                description = "Rename a file or directory.")
    async def rename(self, old_path: str, new_path: str, **kwargs) -> Dict[str, Any]:
        """实现 `rename` 的业务逻辑。"""
        try:
            request = FileMoveRequest(
                src_path=Path(old_path),
                dst_path=Path(new_path)
            )
            result = await self.file_system_service.rename(request)

            extra = result.extra.copy() if result.extra else {}
            extra["old_path"] = old_path
            extra["new_path"] = new_path

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to rename: {str(e)}",
                "extra": {"error": str(e), "old_path": old_path, "new_path": new_path}
            }

    @ecp.action(name = "get_info",
                description = "Get detailed information about a file.")
    async def get_info(self, file_path: str,
                        include_stats: Optional[bool] = True,
                        **kwargs) -> Dict[str, Any]:
        """获取与 `get_info` 对应的数据或状态。"""
        try:
            request = FileStatRequest(path=Path(file_path))
            result = await self.file_system_service.stat(request)

            extra = result.extra.copy() if result.extra else {}
            extra["file_path"] = file_path
            extra["include_stats"] = include_stats

            if result.success and "stats" in extra:
                stats = extra["stats"]
                info = f"File: {file_path}\n"
                info += f"Size: {stats.get('size', 0)} bytes\n"
                info += f"Type: {'Directory' if stats.get('is_directory', False) else 'File'}\n"
                info += f"Permissions: {stats.get('permissions', 'Unknown')}\n"
                if include_stats:
                    info += f"Is Directory: {stats.get('is_directory', False)}\n"
                    info += f"Is File: {stats.get('is_file', False)}\n"
                    info += f"Is Symlink: {stats.get('is_symlink', False)}\n"
                message = info
            else:
                message = result.message

            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get file info: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path}
            }

    @ecp.action(name = "create_dir",
                description = "Create a directory.")
    async def create_dir(self, dir_path: str, **kwargs) -> Dict[str, Any]:
        """创建与 `create_dir` 对应的数据或状态。"""
        try:
            request = DirectoryCreateRequest(path=Path(dir_path))
            result = await self.file_system_service.mkdir(request)

            extra = result.extra.copy() if result.extra else {}
            extra["dir_path"] = dir_path

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create directory: {str(e)}",
                "extra": {"error": str(e), "dir_path": dir_path}
            }

    @ecp.action(name = "delete_dir",
                description = "Delete a directory.")
    async def delete_dir(self, dir_path: str, **kwargs) -> Dict[str, Any]:
        """删除与 `delete_dir` 对应的数据或状态。"""
        try:
            request = DirectoryDeleteRequest(path=Path(dir_path), recursive=True)
            result = await self.file_system_service.rmtree(request)

            extra = result.extra.copy() if result.extra else {}
            extra["dir_path"] = dir_path

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete directory: {str(e)}",
                "extra": {"error": str(e), "dir_path": dir_path}
            }

    @ecp.action(name = "listdir",
                description = "List directory contents.")
    async def listdir(self,
                       dir_path: str,
                       show_hidden: bool = False,
                       file_types: Optional[List[str]] = None,
                       **kwargs) -> Dict[str, Any]:
        """实现 `listdir` 的业务逻辑。"""
        try:
            request = FileListRequest(
                path=Path(dir_path),
                show_hidden=show_hidden,
                file_types=file_types
            )
            result = await self.file_system_service.listdir(request)

            extra = result.extra.copy() if result.extra else {}
            extra["dir_path"] = dir_path

            if result.success:
                if extra.get("files") or extra.get("directories"):
                    listing = f"Contents of {dir_path}:\n"
                    listing += f"Total: {extra.get('total_files', 0)} files, {extra.get('total_directories', 0)} directories\n\n"

                    if extra.get("directories"):
                        listing += "Directories:\n"
                        for directory in extra["directories"]:
                            listing += f"  📁 {directory}/\n"
                        listing += "\n"

                    if extra.get("files"):
                        listing += "Files:\n"
                        for file in extra["files"]:
                            listing += f"  📄 {file}\n"

                    message = listing
                else:
                    message = f"Directory {dir_path} is empty"
            else:
                message = result.message

            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to list directory: {str(e)}",
                "extra": {"error": str(e), "dir_path": dir_path}
            }

    @ecp.action(name = "tree",
                description = "Show directory tree structure.")
    async def tree(self,
                    dir_path: str,
                    max_depth: Optional[int] = 3,
                    show_hidden: bool = False,
                    exclude_patterns: Optional[List[str]] = None,
                    file_types: Optional[List[str]] = None,
                    **kwargs) -> Dict[str, Any]:
        """实现 `tree` 的业务逻辑。"""
        try:
            request = FileTreeRequest(
                path=Path(dir_path),
                max_depth=max_depth,
                show_hidden=show_hidden,
                exclude_patterns=exclude_patterns,
                file_types=file_types
            )
            result = await self.file_system_service.tree(request)

            extra = result.extra.copy() if result.extra else {}
            extra["dir_path"] = dir_path

            if result.success:
                if extra.get("tree_lines"):
                    tree_str = f"Directory tree for {dir_path}:\n"
                    tree_str += "\n".join(extra["tree_lines"])
                    tree_str += f"\n\nTotal: {extra.get('total_files', 0)} files, {extra.get('total_directories', 0)} directories"
                    message = tree_str
                else:
                    message = f"No tree structure found for {dir_path}"
            else:
                message = result.message

            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to generate tree: {str(e)}",
                "extra": {"error": str(e), "dir_path": dir_path}
            }

    @ecp.action(name = "describe",
                description = "Describe the file system with directory structure and file information.")
    async def describe(self, **kwargs) -> Dict[str, Any]:
        """实现 `describe` 的业务逻辑。"""
        try:
            # 处理文件与路径。
            request = FileTreeRequest(
                path=Path("."),
                max_depth=3,
                show_hidden=False
            )
            result = await self.file_system_service.tree(request)

            extra = result.extra.copy() if result.extra else {}
            extra["base_dir"] = str(self.base_dir)

            if result.success:
                description = f"File System Environment at: {self.base_dir}."
                description += f"Total: {extra.get('total_files', 0)} files, {extra.get('total_directories', 0)} directories\n\n"

                if extra.get("tree_lines"):
                    description += "Directory Structure:\n"
                    description += "\n".join(extra["tree_lines"])
                else:
                    description += "No files or directories found."

                message = description
            else:
                message = result.message

            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to describe file system: {str(e)}",
                "extra": {"error": str(e), "base_dir": str(self.base_dir)}
            }

    @ecp.action(name = "search",
                description = "Search for files by name or content.")
    async def search(self,
                      search_path: str,
                      query: str,
                      search_type: str = "name",
                      file_types: Optional[List[str]] = None,
                      case_sensitive: Optional[bool] = False,
                      max_results: Optional[int] = 50,
                      **kwargs) -> Dict[str, Any]:
        """实现 `search` 的业务逻辑。"""
        try:
            request = FileSearchRequest(
                path=Path(search_path),
                query=query,
                by=search_type,
                file_types=file_types,
                case_sensitive=case_sensitive,
                max_results=max_results
            )
            result = await self.file_system_service.search(request)

            extra = result.extra.copy() if result.extra else {}
            extra["search_path"] = search_path
            extra["query"] = query

            if result.success:
                if extra.get("results"):
                    search_str = f"Search results for '{query}' in {search_path}:\n"
                    search_str += f"Found {extra.get('total_found', 0)} results\n\n"

                    results = extra["results"]
                    for i, search_result in enumerate(results, 1):
                        search_str += f"{i}. {search_result.get('path', 'Unknown')}\n"
                        if search_result.get("matches"):
                            for match in search_result["matches"][:5]:  # 说明相关实现细节。
                                search_str += f"   Line {match.get('line', 0)}: {match.get('text', '')[:100]}...\n"
                        search_str += "\n"

                    message = search_str
                else:
                    message = f"No results found for '{query}' in {search_path}"
            else:
                message = result.message

            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to search files: {str(e)}",
                "extra": {"error": str(e), "search_path": search_path, "query": query}
            }

    @ecp.action(name = "change_permissions",
                description = "Change file or directory permissions.")
    async def change_permissions(self, file_path: str, permissions: str, **kwargs) -> Dict[str, Any]:
        """实现 `change_permissions` 的业务逻辑。"""
        try:
            request = FileChangePermissionsRequest(path=Path(file_path), permissions=permissions)
            result = await self.file_system_service.change_permissions(request)

            extra = result.extra.copy() if result.extra else {}
            extra["file_path"] = file_path
            extra["permissions"] = permissions

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to change permissions: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path, "permissions": permissions}
            }

    async def get_state(self, **kwargs) -> Dict[str, Any]:
        """获取与 `get_state` 对应的数据或状态。"""
        try:
            describe_result = await self.describe()
            state_str = dedent(f"""
                <info>
                {describe_result["message"]}
                </info>
            """)
            extra = describe_result["extra"]
            return {
                "state": state_str,
                "extra": extra
            }
        except Exception as e:
            logger.error(f"Failed to get file system state: {e}")
            return {
                "state": f"Failed to get file system state: {str(e)}",
                "extra": {
                    "error": str(e)
                }
            }
