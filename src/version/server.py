"""提供服务入口相关实现。"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import json
import os

from src.logger import logger
from src.config import config
from src.utils import assemble_project_path
from src.utils.file_utils import file_lock
from src.version.types import ComponentVersionHistory

T = TypeVar('T', bound=BaseModel)



class VersionManager(BaseModel):
    """定义 `VersionManager`，封装相关数据与行为。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="The base directory to use for the version histories")
    save_path: str = Field(default=None, description="The path to save version histories")

    def __init__(self, base_dir: Optional[str] = None, save_path: Optional[str] = None, **kwargs):
        """初始化实例。"""
        super().__init__(**kwargs)

        # 处理版本与历史记录。
        self._version_histories: Dict[str, Dict[str, ComponentVersionHistory]] = {
            "tool": {},
            "environment": {},
            "agent": {},
            "prompt": {},
            "memory": {},
            "skill": {}
        }

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "version"))
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "version.json")
        os.makedirs(self.base_dir, exist_ok=True)

        logger.info(f"| 📁 Version manager base directory: {self.base_dir} and save path: {self.save_path}")

    async def initialize(self):
        """初始化组件及其依赖资源。"""
        logger.info(f"| 📁 Version manager initialized.")


    async def register_version(self, component_type: str, name: str, version: str,
                        description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ComponentVersionHistory:
        """注册与 `register_version` 对应的数据或状态。"""
        if component_type not in self._version_histories:
            raise ValueError(f"Unknown component type: {component_type}")

        if name not in self._version_histories[component_type]:
            version_history = ComponentVersionHistory(
                name=name,
                component_type=component_type,
                current_version=version
            )
            self._version_histories[component_type][name] = version_history
        else:
            version_history = self._version_histories[component_type][name]

        version_history.add_version(version, description, metadata)

        # 注册相关组件。
        await self.save_to_json()

        return version_history

    async def list(self) -> Dict[str, Dict[str, List[str]]]:
        """实现 `list` 的业务逻辑。"""
        result = {}
        for component_type, histories in self._version_histories.items():
            result[component_type] = {}
            for name, version_history in histories.items():
                result[component_type][name] = version_history.list_versions()
        return result

    async def get_version_history(self, component_type: str, name: str) -> Optional[ComponentVersionHistory]:
        """获取与 `get_version_history` 对应的数据或状态。"""
        if component_type not in self._version_histories:
            return None

        return self._version_histories[component_type].get(name)

    async def get_current_version(self, component_type: str, name: str) -> Optional[str]:
        """获取与 `get_current_version` 对应的数据或状态。"""
        version_history = await self.get_version_history(component_type, name)
        if version_history is None:
            return None
        return version_history.current_version

    async def generate_next_version(self, component_type: str, name: str,
                                   version_type: str = "patch") -> str:
        """生成与 `generate_next_version` 对应的数据或状态。"""
        current_version = await self.get_current_version(component_type, name)

        if current_version is None:
            # 处理版本与历史记录。
            return "1.0.0"

        try:
            # 转换并规范化数据。
            version_parts = current_version.split(".")
            if len(version_parts) >= 3:
                major = int(version_parts[0])
                minor = int(version_parts[1])
                patch = int(version_parts[2])
            elif len(version_parts) == 2:
                major = int(version_parts[0])
                minor = int(version_parts[1])
                patch = 0
            elif len(version_parts) == 1:
                major = int(version_parts[0])
                minor = 0
                patch = 0
            else:
                # 转换并规范化数据。
                return "1.0.0"

            # 处理版本与历史记录。
            if version_type == "major":
                major += 1
                minor = 0
                patch = 0
            elif version_type == "minor":
                minor += 1
                patch = 0
            else:  # 说明相关实现细节。
                patch += 1

            return f"{major}.{minor}.{patch}"

        except (ValueError, IndexError):
            # 处理异常情况。
            logger.warning(f"| ⚠️ Failed to parse version {current_version} for {component_type}/{name}, starting fresh")
            return "1.0.0"

    async def get_version(self,
                          component_type: str,
                          name: str,
                          provided_version: Optional[str] = None) -> str:
        """获取与 `get_version` 对应的数据或状态。"""
        if provided_version:
            # 处理版本与历史记录。
            return provided_version

        # 加载所需数据。
        current_version = await self.get_current_version(component_type, name)
        if current_version is None:
            # 说明相关实现细节。
            return "1.0.0"
        else:
            # 处理版本与历史记录。
            return await self.generate_next_version(component_type, name, "patch")

    async def deprecate_version(self, component_type: str, name: str, version: str):
        """实现 `deprecate_version` 的业务逻辑。"""
        version_history = await self.get_version_history(component_type, name)
        if version_history is None:
            raise ValueError(f"Component {component_type}/{name} not found")

        version_history.deprecate_version(version)

        # 持久化相关数据。
        await self.save_to_json()

    async def archive_version(self, component_type: str, name: str, version: str):
        """实现 `archive_version` 的业务逻辑。"""
        version_history = await self.get_version_history(component_type, name)
        if version_history is None:
            raise ValueError(f"Component {component_type}/{name} not found")

        version_history.archive_version(version)

        # 持久化相关数据。
        await self.save_to_json()

    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """保存与 `save_to_json` 对应的数据或状态。"""
        file_path = file_path if file_path is not None else self.save_path

        async with file_lock(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 转换并规范化数据。
            save_data = {
                "component_type": {},
                "metadata": {
                    "saved_at": datetime.now().isoformat()
                }
            }

            for component_type, histories in self._version_histories.items():
                save_data["component_type"][component_type] = {}
                for name, version_history in histories.items():
                    # 转换并规范化数据。
                    history_dict = version_history.model_dump(mode="json")
                    save_data["component_type"][component_type][name] = history_dict

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

            logger.info(f"| 💾 Saved version histories to {file_path}")
            return str(file_path)

    async def load_from_json(self, file_path: Optional[str] = None) -> bool:
        """加载与 `load_from_json` 对应的数据或状态。"""
        file_path = file_path if file_path is not None else self.save_path

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Version file not found: {file_path}")
                return False

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)

                # 说明相关实现细节。
                for component_type in self._version_histories:
                    self._version_histories[component_type].clear()

                # 加载所需数据。
                component_types = load_data.get("component_type", {})
                for component_type, histories in component_types.items():
                    if component_type not in self._version_histories:
                        logger.warning(f"| ⚠️ Unknown component type: {component_type}")
                        continue

                    for name, history_dict in histories.items():
                        try:
                            # 处理版本与历史记录。
                            version_history = ComponentVersionHistory(**history_dict)
                            self._version_histories[component_type][name] = version_history
                        except Exception as e:
                            logger.error(f"| ❌ Failed to load version history for {name}: {e}")
                            continue

                logger.info(f"| 📂 Loaded version histories from {file_path}")
                return True

            except Exception as e:
                logger.error(f"| ❌ Failed to load version data from {file_path}: {e}")
                return False

    @staticmethod
    def compare_versions(v1: str, v2: str) -> int:
        """实现 `compare_versions` 的业务逻辑。"""
        try:
            parts1 = [int(x) for x in v1.split(".")]
            parts2 = [int(x) for x in v2.split(".")]

            # 说明相关实现细节。
            max_len = max(len(parts1), len(parts2))
            parts1.extend([0] * (max_len - len(parts1)))
            parts2.extend([0] * (max_len - len(parts2)))

            for p1, p2 in zip(parts1, parts2):
                if p1 > p2:
                    return 1
                elif p1 < p2:
                    return -1
            return 0
        except:
            # 执行回退或重试逻辑。
            return 1 if v1 > v2 else (-1 if v1 < v2 else 0)


# 处理版本与历史记录。
version_manager = VersionManager()
