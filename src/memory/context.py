"""提供上下文管理相关实现。"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import inflection
from asyncio_atexit import register as async_atexit_register
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.optimizer.types import Variable

import builtins

from src.config import config
from src.dynamic import dynamic_manager
from src.logger import logger
from src.memory.types import Memory, MemoryConfig
from src.registry import MEMORY_SYSTEM, load_builtin_components
from src.session import SessionContext
from src.utils import (
    assemble_project_path,
    file_lock,
    gather_with_concurrency,
    read_json_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from src.version import version_manager


class MemoryContextManager(BaseModel):
    """定义 `MemoryContextManager`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(
        default=None, description="The base directory to use for the memory systems"
    )
    save_path: str = Field(
        default=None, description="The path to save the memory systems"
    )
    contract_path: str = Field(
        default=None, description="The path to save the memory contract"
    )

    def __init__(
        self,
        base_dir: str | None = None,
        save_path: str | None = None,
        contract_path: str | None = None,
        **kwargs,
    ):
        """初始化实例。"""
        super().__init__(**kwargs)

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(
                os.path.join(config.workdir, "memory")
            )
        logger.info(f"| 📁 Memory context manager base directory: {self.base_dir}.")
        os.makedirs(self.base_dir, exist_ok=True)
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "memory.json")
        logger.info(f"| 📁 Memory context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Memory context manager contract path: {self.contract_path}.")

        self._memory_configs: dict[str, MemoryConfig] = {}  # 配置相关参数。
        # 配置相关参数。
        self._memory_history_versions: dict[str, dict[str, MemoryConfig]] = {}

        self._cleanup_registered = False
        self._variables_lock = asyncio.Lock()  # 更新相关状态。

    async def initialize(self, memory_names: list[str] | None = None):
        """初始化组件及其依赖资源。"""
        # 注册相关组件。
        dynamic_manager.register_symbol("MEMORY_SYSTEM", MEMORY_SYSTEM)
        dynamic_manager.register_symbol("Memory", Memory)

        # 注册相关组件。
        def memory_context_provider():
            """实现 `memory_context_provider` 的业务逻辑。"""
            return {
                "MEMORY_SYSTEM": MEMORY_SYSTEM,
                "Memory": Memory,
            }

        dynamic_manager.register_context_provider("memory", memory_context_provider)

        # 加载所需数据。
        memory_configs = {}
        registry_memory_configs: dict[
            str, MemoryConfig
        ] = await self._load_from_registry()
        memory_configs.update(registry_memory_configs)

        # 加载所需数据。
        code_memory_configs: dict[str, MemoryConfig] = await self._load_from_code()

        # 配置相关参数。
        for memory_name, code_config in code_memory_configs.items():
            if memory_name in memory_configs:
                registry_config = memory_configs[memory_name]
                # 处理版本与历史记录。
                if (
                    version_manager.compare_versions(
                        code_config.version, registry_config.version
                    )
                    > 0
                ):
                    logger.info(
                        f"| 🔄 Overriding memory {memory_name} from registry (v{registry_config.version}) with code version (v{code_config.version})"
                    )
                    memory_configs[memory_name] = code_config
                else:
                    logger.info(
                        f"| 📌 Keeping memory {memory_name} from registry (v{registry_config.version}), code version (v{code_config.version}) is not greater"
                    )
                    # 配置相关参数。
                    if (
                        version_manager.compare_versions(
                            code_config.version, registry_config.version
                        )
                        == 0
                        and memory_name in self._memory_history_versions
                    ):
                        self._memory_history_versions[memory_name][
                            registry_config.version
                        ] = registry_config
            else:
                # 处理记忆或缓存状态。
                memory_configs[memory_name] = code_config

        # 处理记忆或缓存状态。
        if memory_names is not None:
            memory_configs = {
                name: memory_configs[name]
                for name in memory_names
                if name in memory_configs
            }

        # 创建所需对象。
        memory_names_list = list(memory_configs.keys())
        tasks = [self.build(memory_configs[name]) for name in memory_names_list]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )

        for memory_name, result in zip(memory_names_list, results):
            if isinstance(result, Exception):
                logger.error(
                    f"| ❌ Failed to initialize memory {memory_name}: {result}"
                )
                continue
            self._memory_configs[memory_name] = result
            logger.info(f"| 🔧 Memory {memory_name} initialized")

        # 配置相关参数。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract(memory_names=memory_names_list)

        # 清理并释放相关资源。
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True

        logger.info("| ✅ Memory systems initialization completed")

    async def _load_from_registry(self):
        """实现 `_load_from_registry` 的业务逻辑。"""

        memory_configs: dict[str, MemoryConfig] = {}

        async def register_memory_class(memory_cls: type[Memory]):
            """注册与 `register_memory_class` 对应的数据或状态。"""
            try:
                # 配置相关参数。
                memory_config_key = inflection.underscore(memory_cls.__name__)
                memory_config_dict = config.get(memory_config_key, {})
                memory_require_grad = (
                    memory_config_dict.get("require_grad", False)
                    if memory_config_dict and "require_grad" in memory_config_dict
                    else False
                )

                # 创建所需对象。
                try:
                    temp_instance = memory_cls(**memory_config_dict)
                    memory_name = temp_instance.name
                    memory_description = temp_instance.description
                except Exception:
                    # 配置相关参数。
                    try:
                        temp_instance = memory_cls()
                        memory_name = temp_instance.name
                        memory_description = temp_instance.description
                    except Exception:
                        # 处理异常情况。
                        memory_name = getattr(memory_cls, "name", None)
                        memory_description = getattr(memory_cls, "description", "")
                        if not memory_name:
                            # 执行回退或重试逻辑。
                            memory_name = inflection.underscore(memory_cls.__name__)
                        if not memory_description:
                            memory_description = memory_cls.__doc__ or ""

                # 处理版本与历史记录。
                memory_version = await version_manager.get_version(
                    "memory", memory_name
                )

                # 说明相关实现细节。
                memory_code = dynamic_manager.get_full_module_source(memory_cls)

                # 配置相关参数。
                memory_config = MemoryConfig(
                    name=memory_name,
                    description=memory_description,
                    require_grad=memory_require_grad,
                    version=memory_version,
                    cls=memory_cls,
                    config=memory_config_dict,
                    instance=None,
                    metadata={},
                    code=memory_code,
                )

                # 配置相关参数。
                memory_configs[memory_name] = memory_config

                # 处理版本与历史记录。
                if memory_name not in self._memory_history_versions:
                    self._memory_history_versions[memory_name] = {}
                self._memory_history_versions[memory_name][memory_version] = (
                    memory_config
                )

                # 注册相关组件。
                await version_manager.register_version(
                    "memory", memory_name, memory_version
                )

                logger.info(
                    f"| 📝 Registered memory: {memory_name} ({memory_cls.__name__})"
                )

            except Exception as e:
                logger.error(
                    f"| ❌ Failed to register memory class {memory_cls.__name__}: {e}"
                )
                raise

        load_builtin_components("memory")

        # 注册相关组件。
        memory_classes = list(MEMORY_SYSTEM._module_dict.values())

        logger.info(
            f"| 🔍 Discovering {len(memory_classes)} memory systems from MEMORY_SYSTEM registry"
        )

        # 注册相关组件。
        tasks = [register_memory_class(memory_cls) for memory_cls in memory_classes]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )
        success_count = sum(1 for r in results if not isinstance(r, Exception))

        logger.info(
            f"| ✅ Discovered and registered {success_count}/{len(memory_classes)} memory systems from MEMORY_SYSTEM registry"
        )

        return memory_configs

    async def _load_from_code(self):
        """实现 `_load_from_code` 的业务逻辑。"""
        memory_configs: dict[str, MemoryConfig] = {}

        # 加载所需数据。
        if not os.path.exists(self.save_path):
            logger.info(
                f"| 📂 Memory config file not found at {self.save_path}, skipping code-based loading"
            )
            return memory_configs

        # 配置相关参数。
        try:
            load_data = await read_json_file(self.save_path)
        except json.JSONDecodeError as e:
            logger.warning(
                f"| ⚠️ Failed to parse memory config JSON from {self.save_path}: {e}"
            )
            return memory_configs

        memories_data = load_data.get("memory_systems", {})

        async def register_memory_class(
            memory_name: str, memory_data: dict[str, Any]
        ) -> tuple[str, dict[str, MemoryConfig], MemoryConfig | None] | None:
            """注册与 `register_memory_class` 对应的数据或状态。"""
            try:
                current_version = memory_data.get("current_version", "1.0.0")
                versions = memory_data.get("versions", {})

                if not versions:
                    logger.warning(f"| ⚠️ Memory {memory_name} has no versions")
                    return None

                version_map: dict[str, MemoryConfig] = {}
                current_config: MemoryConfig | None = None  # 配置相关参数。

                for version_data in versions.values():
                    # 配置相关参数。
                    memory_config = MemoryConfig.from_dict(version_data)
                    version = memory_config.version
                    version_map[version] = memory_config

                    if version == current_version:
                        current_config = memory_config

                return memory_name, version_map, current_config
            except Exception as e:
                logger.error(
                    f"| ❌ Failed to load memory {memory_name} from code JSON: {e}"
                )
                return None

        # 加载所需数据。
        tasks = [
            register_memory_class(memory_name, memory_data)
            for memory_name, memory_data in memories_data.items()
        ]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            memory_name, version_map, current_config = result
            if not version_map:
                continue

            # 处理版本与历史记录。
            self._memory_history_versions[memory_name] = version_map
            # 配置相关参数。
            if current_config is not None:
                memory_configs[memory_name] = current_config
            else:
                # 执行回退或重试逻辑。
                logger.warning(
                    f"| ⚠️ Memory {memory_name} current_version not found, using last available version"
                )
                memory_configs[memory_name] = list(version_map.values())[-1]

            # 注册相关组件。
            for memory_config in version_map.values():
                await version_manager.register_version(
                    "memory", memory_name, memory_config.version
                )

        logger.info(
            f"| 📂 Loaded {len(memory_configs)} memory systems from {self.save_path}"
        )
        return memory_configs

    async def register(
        self,
        memory: Memory | type[Memory],
        memory_config_dict: dict[str, Any] | None = None,
        override: bool = False,
        version: str | None = None,
    ) -> MemoryConfig:
        """实现 `register` 的业务逻辑。"""

        try:
            # 说明相关实现细节。
            if isinstance(memory, Memory):
                # 注册相关组件。
                memory_instance = memory
                memory_cls = type(memory)
                if memory_config_dict:
                    raise ValueError(
                        "Extra keyword arguments are not allowed when registering memory instances."
                    )
                memory_config_dict = {}
            else:
                # 注册相关组件。
                memory_cls = memory
                if memory_config_dict is None:
                    # 配置相关参数。
                    memory_config_key = inflection.underscore(memory_cls.__name__)
                    memory_config_dict = config.get(memory_config_key, {})

                # 注册相关组件。
                try:
                    memory_instance = memory_cls(**memory_config_dict)
                except Exception as e:
                    logger.error(
                        f"| ❌ Failed to create memory instance for {memory_cls.__name__}: {e}"
                    )
                    raise ValueError(
                        f"Failed to instantiate memory {memory_cls.__name__} with provided config: {e}"
                    )

            memory_name = memory_instance.name
            memory_description = memory_instance.description
            memory_metadata = getattr(memory_instance, "metadata", {})
            # 配置相关参数。
            memory_require_grad = (
                memory_config_dict.get("require_grad", memory_instance.require_grad)
                if memory_config_dict and "require_grad" in memory_config_dict
                else memory_instance.require_grad
            )

            if not memory_name:
                raise ValueError("Memory.name cannot be empty.")

            if memory_name in self._memory_configs and not override:
                raise ValueError(
                    f"Memory '{memory_name}' already registered. Use override=True to replace it."
                )

            # 处理版本与历史记录。
            if version is None:
                memory_version = await version_manager.get_version(
                    "memory", memory_name
                )
            else:
                memory_version = version

            # 处理记忆或缓存状态。
            memory_code = dynamic_manager.get_full_module_source(memory_cls)
            if not memory_code:
                logger.warning(
                    f"| ⚠️ Memory {memory_name} source code cannot be extracted"
                )

            # 配置相关参数。
            memory_config = MemoryConfig(
                name=memory_name,
                description=memory_description,
                require_grad=memory_require_grad,
                version=memory_version,
                cls=memory_cls,
                config=memory_config_dict or {},
                instance=memory_instance if isinstance(memory, Memory) else None,
                metadata=memory_metadata,
                code=memory_code,
            )

            # 配置相关参数。
            self._memory_configs[memory_name] = memory_config

            # 处理版本与历史记录。
            if memory_name not in self._memory_history_versions:
                self._memory_history_versions[memory_name] = {}
            self._memory_history_versions[memory_name][memory_config.version] = (
                memory_config
            )

            # 注册相关组件。
            await version_manager.register_version(
                "memory", memory_name, memory_config.version
            )

            # 持久化相关数据。
            await self.save_to_json()
            # 持久化相关数据。
            await self.save_contract()

            logger.info(
                f"| 📝 Registered memory config: {memory_name}: {memory_config.version}"
            )
            return memory_config

        except Exception as e:
            logger.error(f"| ❌ Failed to register memory: {e}")
            raise

    async def update(
        self,
        memory_name: str,
        memory: Memory | type[Memory],
        memory_config_dict: dict[str, Any] | None = None,
        new_version: str | None = None,
        description: str | None = None,
        code: str | None = None,
    ) -> MemoryConfig:
        """实现 `update` 的业务逻辑。"""
        try:
            # 说明相关实现细节。
            if isinstance(memory, Memory):
                # 说明相关实现细节。
                memory_instance = memory
                memory_cls = type(memory)
                if memory_config_dict:
                    raise ValueError(
                        "Extra keyword arguments are not allowed when updating with memory instances."
                    )
                memory_config_dict = {}
            else:
                # 说明相关实现细节。
                memory_cls = memory
                if memory_config_dict is None:
                    # 配置相关参数。
                    memory_config_key = inflection.underscore(memory_cls.__name__)
                    memory_config_dict = config.get(memory_config_key, {})

                # 更新相关状态。
                try:
                    memory_instance = memory_cls(**memory_config_dict)
                except Exception as e:
                    logger.error(
                        f"| ❌ Failed to create memory instance for {memory_cls.__name__}: {e}"
                    )
                    raise ValueError(
                        f"Failed to instantiate memory {memory_cls.__name__} with provided config: {e}"
                    )

            # 校验输入与当前状态。
            original_config = self._memory_configs.get(memory_name)
            if original_config is None:
                raise ValueError(
                    f"Memory {memory_name} not found. Use register() to register a new memory system."
                )

            memory_description = memory_instance.description
            memory_metadata = getattr(memory_instance, "metadata", {})
            # 配置相关参数。
            memory_require_grad = (
                memory_config_dict.get("require_grad", memory_instance.require_grad)
                if memory_config_dict and "require_grad" in memory_config_dict
                else memory_instance.require_grad
            )

            # 处理版本与历史记录。
            if new_version is None:
                # 处理版本与历史记录。
                new_version = await version_manager.generate_next_version(
                    "memory", memory_name, "patch"
                )

            # 创建所需对象。
            if code is not None:
                memory_code = code
            else:
                memory_code = dynamic_manager.get_full_module_source(memory_cls)
                if not memory_code:
                    logger.warning(
                        f"| ⚠️ Memory {memory_name} source code cannot be extracted"
                    )

            # 配置相关参数。
            updated_config = MemoryConfig(
                name=memory_name,  # 说明相关实现细节。
                description=memory_description,
                require_grad=memory_require_grad,
                version=new_version,
                cls=memory_cls,
                config=memory_config_dict or {},
                instance=memory_instance,  # 创建所需对象。
                metadata=memory_metadata,
                code=memory_code,
            )

            # 配置相关参数。
            self._memory_configs[memory_name] = updated_config

            # 处理版本与历史记录。
            if memory_name not in self._memory_history_versions:
                self._memory_history_versions[memory_name] = {}
            self._memory_history_versions[memory_name][updated_config.version] = (
                updated_config
            )

            # 注册相关组件。
            await version_manager.register_version(
                "memory",
                memory_name,
                new_version,
                description=description or f"Updated from {original_config.version}",
            )

            # 持久化相关数据。
            await self.save_to_json()
            # 持久化相关数据。
            await self.save_contract()

            logger.info(
                f"| 🔄 Updated memory {memory_name} from v{original_config.version} to v{new_version}"
            )
            return updated_config

        except Exception as e:
            logger.error(f"| ❌ Failed to update memory: {e}")
            raise

    async def copy(
        self,
        memory_name: str,
        new_name: str | None = None,
        new_version: str | None = None,
        new_config: dict[str, Any] | None = None,
    ) -> MemoryConfig:
        """实现 `copy` 的业务逻辑。"""
        try:
            original_config = self._memory_configs.get(memory_name)
            if original_config is None:
                raise ValueError(f"Memory {memory_name} not found")

            if original_config.cls is None:
                raise ValueError(f"Cannot copy memory {memory_name}: no class provided")

            # 说明相关实现细节。
            if new_name is None:
                new_name = memory_name

            # 配置相关参数。
            memory_config_dict = (
                original_config.config.copy() if original_config.config else {}
            )
            if new_config:
                # 配置相关参数。
                memory_config_dict.update(new_config)

            # 处理记忆或缓存状态。
            try:
                memory_instance = original_config.cls(**memory_config_dict)
            except Exception as e:
                logger.error(
                    f"| ❌ Failed to create memory instance for {original_config.cls.__name__}: {e}"
                )
                raise ValueError(
                    f"Failed to instantiate memory {original_config.cls.__name__} with provided config: {e}"
                )

            # 说明相关实现细节。
            if new_name != memory_name:
                memory_instance.name = new_name

            memory_description = memory_instance.description
            memory_metadata = getattr(memory_instance, "metadata", {})
            memory_require_grad = (
                memory_config_dict.get("require_grad", memory_instance.require_grad)
                if memory_config_dict and "require_grad" in memory_config_dict
                else memory_instance.require_grad
            )

            # 处理版本与历史记录。
            if new_version is None:
                if new_name == memory_name:
                    # 处理版本与历史记录。
                    new_version = await version_manager.generate_next_version(
                        "memory", new_name, "patch"
                    )
                else:
                    # 处理版本与历史记录。
                    new_version = await version_manager.get_version("memory", new_name)

            # 处理记忆或缓存状态。
            memory_code = dynamic_manager.get_full_module_source(original_config.cls)
            if not memory_code:
                logger.warning(f"| ⚠️ Memory {new_name} source code cannot be extracted")

            # 配置相关参数。
            new_memory_config = MemoryConfig(
                name=new_name,
                description=memory_description,
                require_grad=memory_require_grad,
                version=new_version,
                cls=original_config.cls,
                config=memory_config_dict,
                instance=memory_instance,
                metadata=memory_metadata,
                code=memory_code,
            )

            # 注册相关组件。
            self._memory_configs[new_name] = new_memory_config

            # 处理版本与历史记录。
            if new_name not in self._memory_history_versions:
                self._memory_history_versions[new_name] = {}
            self._memory_history_versions[new_name][new_version] = new_memory_config

            # 注册相关组件。
            await version_manager.register_version(
                "memory",
                new_name,
                new_version,
                description=f"Copied from {memory_name}@{original_config.version}",
            )

            # 持久化相关数据。
            await self.save_to_json()
            # 持久化相关数据。
            await self.save_contract()

            logger.info(
                f"| 📋 Copied memory {memory_name}@{original_config.version} to {new_name}@{new_version}"
            )
            return new_memory_config

        except Exception as e:
            logger.error(f"| ❌ Failed to copy memory: {e}")
            raise

    async def unregister(self, memory_name: str) -> bool:
        """实现 `unregister` 的业务逻辑。"""
        if memory_name not in self._memory_configs:
            logger.warning(f"| ⚠️ Memory {memory_name} not found")
            return False

        memory_config = self._memory_configs[memory_name]

        # 配置相关参数。
        del self._memory_configs[memory_name]

        # 注册相关组件。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract()

        logger.info(f"| 🗑️ Unregistered memory {memory_name}@{memory_config.version}")
        return True

    async def get(self, memory_name: str) -> Memory:
        """实现 `get` 的业务逻辑。"""
        memory_config = self._memory_configs.get(memory_name)
        if memory_config is None:
            return None
        return memory_config.instance if memory_config.instance is not None else None

    async def get_info(self, memory_name: str) -> MemoryConfig | None:
        """获取与 `get_info` 对应的数据或状态。"""
        return self._memory_configs.get(memory_name)

    async def list(self) -> list[str]:
        """实现 `list` 的业务逻辑。"""
        return [name for name in self._memory_configs]

    async def build(self, memory_config: MemoryConfig) -> MemoryConfig:
        """实现 `build` 的业务逻辑。"""
        if memory_config.name in self._memory_configs:
            existing_config = self._memory_configs[memory_config.name]
            if existing_config.instance is not None:
                return existing_config

        # 创建所需对象。
        try:
            # 加载所需数据。
            if memory_config.cls is None:
                raise ValueError(
                    f"Cannot create memory {memory_config.name}: no class provided. Class should be loaded during initialization."
                )

            # 处理记忆或缓存状态。
            memory_instance = (
                memory_config.cls(**memory_config.config)
                if memory_config.config
                else memory_config.cls()
            )

            # 初始化相关状态。
            if hasattr(memory_instance, "initialize"):
                await memory_instance.initialize()

            memory_config.instance = memory_instance

            # 处理记忆或缓存状态。
            self._memory_configs[memory_config.name] = memory_config

            logger.info(f"| 🔧 Memory {memory_config.name} created and stored")

            return memory_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create memory {memory_config.name}: {e}")
            raise

    async def save_to_json(self, file_path: str | None = None) -> str:
        """保存与 `save_to_json` 对应的数据或状态。"""
        file_path = file_path if file_path is not None else self.save_path

        async with file_lock(file_path):
            # 处理文件与路径。
            parent_dir = os.path.dirname(file_path)
            if parent_dir:  # 创建所需对象。
                os.makedirs(parent_dir, exist_ok=True)

            # 持久化相关数据。
            save_data = {
                "metadata": {
                    "saved_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "num_memories": len(self._memory_configs),
                    "num_versions": sum(
                        len(versions)
                        for versions in self._memory_history_versions.values()
                    ),
                },
                "memory_systems": {},
            }

            for memory_name, version_map in self._memory_history_versions.items():
                try:
                    versions_data: dict[str, dict[str, Any]] = {}
                    for memory_config in version_map.values():
                        config_dict = memory_config.model_dump()
                        versions_data[memory_config.version] = config_dict

                    # 配置相关参数。
                    # 配置相关参数。
                    current_version = None
                    if memory_name in self._memory_configs:
                        current_config = self._memory_configs[memory_name]
                        if current_config is not None:
                            current_version = current_config.version

                    # 配置相关参数。
                    if current_version is None and version_map:
                        # 处理版本与历史记录。
                        latest_version_str = None
                        for version_str in version_map:
                            if (
                                latest_version_str is None
                                or version_manager.compare_versions(
                                    version_str, latest_version_str
                                )
                                > 0
                            ):
                                latest_version_str = version_str
                        current_version = latest_version_str

                    save_data["memory_systems"][memory_name] = {
                        "versions": versions_data,
                        "current_version": current_version,
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize memory {memory_name}: {e}")
                    continue

            # 持久化相关数据。
            await write_json_file(file_path, save_data)

            logger.info(
                f"| 💾 Saved {len(self._memory_configs)} memory systems with version history to {file_path}"
            )
            return str(file_path)

    async def save_contract(self, memory_names: builtins.list[str] | None = None):
        """保存与 `save_contract` 对应的数据或状态。"""
        contract = []
        if memory_names is not None:
            for index, memory_name in enumerate(memory_names):
                memory_info = await self.get_info(memory_name)
                if memory_info:
                    text = f"Name: {memory_info.name}\nDescription: {memory_info.description}\nRequire Grad: {memory_info.require_grad}"
                    contract.append(f"{index + 1:04d}\n{text}\n")
        else:
            for index, memory_name in enumerate(self._memory_configs.keys()):
                memory_info = await self.get_info(memory_name)
                if memory_info:
                    text = f"Name: {memory_info.name}\nDescription: {memory_info.description}\nRequire Grad: {memory_info.require_grad}"
                    contract.append(f"{index + 1:04d}\n{text}\n")
        contract_text = "---\n".join(contract)
        await write_text_file(self.contract_path, contract_text)
        logger.info(
            f"| 📝 Saved {len(contract)} memory systems contract to {self.contract_path}"
        )

    async def load_contract(self) -> str:
        """加载与 `load_contract` 对应的数据或状态。"""
        if not os.path.exists(self.contract_path):
            return ""
        return await read_text_file(self.contract_path)

    async def load_from_json(
        self, file_path: str | None = None, auto_initialize: bool = True
    ) -> bool:
        """加载与 `load_from_json` 对应的数据或状态。"""

        file_path = file_path if file_path is not None else self.save_path

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Memory file not found: {file_path}")
                return False

            try:
                load_data = await read_json_file(file_path)

                memories_data = load_data.get("memory_systems", {})
                loaded_count = 0

                for memory_name, memory_data in memories_data.items():
                    try:
                        # 配置相关参数。
                        versions_data = memory_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(
                                f"| ⚠️ Memory {memory_name} has invalid format for 'versions' (expected dict), skipping"
                            )
                            continue

                        current_version_str = memory_data.get("current_version")

                        # 加载所需数据。
                        version_configs = []
                        latest_config = None
                        latest_version = None

                        for version_str, config_dict in versions_data.items():
                            # 处理版本与历史记录。
                            if "version" not in config_dict:
                                config_dict["version"] = version_str

                            try:
                                memory_config = MemoryConfig.from_dict(config_dict)
                                version_configs.append(memory_config)
                            except Exception as e:
                                logger.warning(
                                    f"| ⚠️ Failed to load memory config for {memory_name}@{version_str}: {e}"
                                )
                                continue

                            # 处理版本与历史记录。
                            if (
                                latest_config is None
                                or (
                                    current_version_str
                                    and memory_config.version == current_version_str
                                )
                                or (
                                    not current_version_str
                                    and (
                                        latest_version is None
                                        or version_manager.compare_versions(
                                            memory_config.version, latest_version
                                        )
                                        > 0
                                    )
                                )
                            ):
                                latest_config = memory_config
                                latest_version = memory_config.version

                        # 处理版本与历史记录。
                        self._memory_history_versions[memory_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }

                        # 更新相关状态。
                        if latest_config:
                            self._memory_configs[memory_name] = latest_config

                            # 注册相关组件。
                            for memory_config in version_configs:
                                await version_manager.register_version(
                                    "memory", memory_name, memory_config.version
                                )

                            # 持久化相关数据。
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)

                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load memory {memory_name}: {e}")
                        continue

                logger.info(
                    f"| 📂 Loaded {loaded_count} memory systems with version history from {file_path}"
                )
                return True

            except Exception as e:
                logger.error(
                    f"| ❌ Failed to load memory systems from {file_path}: {e}"
                )
                return False

    async def restore(
        self, memory_name: str, version: str, auto_initialize: bool = True
    ) -> MemoryConfig | None:
        """实现 `restore` 的业务逻辑。"""
        # 处理版本与历史记录。
        version_config = None
        if memory_name in self._memory_history_versions:
            version_config = self._memory_history_versions[memory_name].get(version)

        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for memory {memory_name}")
            return None

        # 创建所需对象。
        restored_config = MemoryConfig(**version_config.model_dump())

        # 配置相关参数。
        self._memory_configs[memory_name] = restored_config

        # 更新相关状态。
        version_history = await version_manager.get_version_history(
            "memory", memory_name
        )
        if version_history:
            # 注册相关组件。
            if version not in version_history.versions:
                await version_manager.register_version("memory", memory_name, version)
            version_history.current_version = version
        else:
            # 注册相关组件。
            await version_manager.register_version("memory", memory_name, version)

        # 初始化相关状态。
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)

        # 持久化相关数据。
        await self.save_to_json()

        logger.info(f"| 🔄 Restored memory {memory_name} to version {version}")
        return restored_config

    async def get_variables(
        self, memory_name: str | None = None
    ) -> dict[str, "Variable"]:
        """获取与 `get_variables` 对应的数据或状态。"""
        # 说明相关实现细节。
        from src.optimizer.types import Variable

        variables: dict[str, Variable] = {}

        if memory_name is not None:
            # 处理记忆或缓存状态。
            memory_config = self._memory_configs.get(memory_name)
            if memory_config is None:
                logger.warning(f"| ⚠️ Memory {memory_name} not found")
                return variables

            memory_configs = {memory_name: memory_config}
        else:
            # 处理记忆或缓存状态。
            memory_configs = self._memory_configs

        for name, memory_config in memory_configs.items():
            # 处理记忆或缓存状态。
            memory_code = memory_config.code or ""

            # 创建所需对象。
            variable = Variable(
                name=name,
                type="memory_code",
                description=memory_config.description
                or f"Code for memory system {name}",
                require_grad=memory_config.require_grad,
                template=None,
                variables=memory_code,  # 说明相关实现细节。
            )
            variables[name] = variable

        return variables

    async def get_trainable_variables(
        self, memory_name: str | None = None
    ) -> dict[str, "Variable"]:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        async with self._variables_lock:
            # 说明相关实现细节。
            all_variables = await self.get_variables(memory_name=memory_name)

            # 说明相关实现细节。
            trainable_variables = {
                name: variable
                for name, variable in all_variables.items()
                if variable.require_grad is True
            }

            return trainable_variables

    async def set_variables(
        self,
        memory_name: str,
        variable_updates: dict[str, Any],
        new_version: str | None = None,
        description: str | None = None,
    ) -> MemoryConfig:
        """设置与 `set_variables` 对应的数据或状态。"""
        async with self._variables_lock:
            original_config = self._memory_configs.get(memory_name)
            if original_config is None:
                raise ValueError(
                    f"Memory {memory_name} not found. Use register() to register a new memory system."
                )

            # 更新相关状态。
            # 说明相关实现细节。
            if "variables" not in variable_updates:
                raise ValueError(
                    f"variable_updates must contain 'variables' field with memory code, got: {list(variable_updates.keys())}"
                )

            new_code = variable_updates["variables"]
            if not isinstance(new_code, str):
                raise TypeError(f"Memory code must be a string, got {type(new_code)}")

            # 加载所需数据。
            class_name = dynamic_manager.extract_class_name_from_code(new_code)
            if not class_name:
                raise ValueError("Cannot extract class name from code")

            try:
                memory_cls = dynamic_manager.load_class(
                    new_code, class_name=class_name, base_class=Memory, context="memory"
                )
            except Exception as e:
                logger.error(f"| ❌ Failed to load memory class from code: {e}")
                raise ValueError(f"Failed to load memory class from code: {e}")

            # 持久化相关数据。
            # 创建所需对象。
            update_description = description or f"Updated code for {memory_name}"
            return await self.update(
                memory_name=memory_name,
                memory=memory_cls,
                memory_config_dict=original_config.config,
                new_version=new_version,
                description=update_description,
                code=new_code,  # 创建所需对象。
            )

    async def cleanup(self):
        """释放组件占用的资源。"""
        try:
            # 配置相关参数。
            self._memory_configs.clear()
            self._memory_history_versions.clear()

            logger.info("| 🧹 Memory context manager cleaned up")

        except Exception as e:
            logger.error(f"| ❌ Error during memory context manager cleanup: {e}")

    async def start_session(
        self,
        memory_name: str,
        agent_name: str | None = None,
        task_id: str | None = None,
        description: str | None = None,
        ctx: SessionContext | None = None,
        **kwargs,
    ) -> str:
        """实现 `start_session` 的业务逻辑。"""
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
        return await instance.start_session(
            agent_name=agent_name,
            task_id=task_id,
            description=description,
            ctx=ctx,
            **kwargs,
        )

    async def add_event(
        self,
        memory_name: str,
        step_number: int,
        event_type: Any,
        data: Any,
        agent_name: str,
        task_id: str | None = None,
        ctx: SessionContext = None,
        **kwargs,
    ):
        """添加与 `add_event` 对应的数据或状态。"""
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
        return await instance.add_event(
            step_number, event_type, data, agent_name, task_id, ctx=ctx, **kwargs
        )

    async def end_session(self, memory_name: str, ctx: SessionContext = None, **kwargs):
        """实现 `end_session` 的业务逻辑。"""
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
        return await instance.end_session(ctx=ctx, **kwargs)

    async def get_session_info(
        self, memory_name: str, ctx: SessionContext = None, **kwargs
    ):
        """获取与 `get_session_info` 对应的数据或状态。"""
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
        return await instance.get_session_info(ctx=ctx, **kwargs)

    async def clear_session(
        self, memory_name: str, ctx: SessionContext = None, **kwargs
    ):
        """实现 `clear_session` 的业务逻辑。"""
        instance = await self.get(memory_name)
        if instance is None:
            raise ValueError(f"Memory system '{memory_name}' not found")
        return await instance.clear_session(ctx=ctx, **kwargs)

    async def get_state(
        self,
        memory_name: str,
        n: int | None = None,
        ctx: SessionContext | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """获取与 `get_state` 对应的数据或状态。"""
        memory_info = await self.get_info(memory_name)
        if memory_info is None or memory_info.instance is None:
            raise ValueError(f"记忆系统不存在或尚未初始化：{memory_name}")

        version = memory_info.version
        memory_instance = memory_info.instance
        logger.info(f"| ✅ Using memory {memory_name}@{version}")

        # 处理记忆或缓存状态。
        events = await memory_instance.get_event(n=n, ctx=ctx, **kwargs)
        summaries = await memory_instance.get_summary(n=n, ctx=ctx, **kwargs)
        insights = await memory_instance.get_insight(n=n, ctx=ctx, **kwargs)

        return {"events": events, "summaries": summaries, "insights": insights}
