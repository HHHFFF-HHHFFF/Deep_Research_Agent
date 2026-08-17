"""提供上下文管理相关实现。"""
import os
import asyncio
from asyncio_atexit import register as async_atexit_register
from typing import Any, Dict, List, Type, Optional, Union, Tuple, TYPE_CHECKING
from datetime import datetime
import inflection
import json
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from src.logger import logger
from src.config import config
from src.environment.faiss.service import FaissService
from src.environment.faiss.types import FaissAddRequest
from src.utils import (assemble_project_path,
                       gather_with_concurrency,
                       file_lock
                       )
from src.tool.types import Tool, ToolConfig, ToolResponse
from src.session import SessionContext
from src.version import version_manager
from src.dynamic import dynamic_manager
from src.registry import TOOL

class ToolContextManager(BaseModel):
    """定义 `ToolContextManager`，封装相关数据与行为。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="The base directory to use for the tools")
    save_path: str = Field(default=None, description="The path to save the tools")
    contract_path: str = Field(default=None, description="The path to save the tool contract")

    def __init__(self,
                 base_dir: Optional[str] = None,
                 save_path: Optional[str] = None,
                 contract_path: Optional[str] = None,
                 model_name: str = "openrouter/gemini-3-flash-preview",
                 embedding_model_name: str = "openrouter/text-embedding-3-large",
                 default_timeout: Optional[float] = 1800.0,
                 **kwargs):
        """初始化实例。"""
        super().__init__(**kwargs)

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "tool"))
        logger.info(f"| 📁 Tool context manager base directory: {self.base_dir}.")
        os.makedirs(self.base_dir, exist_ok=True)
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "tool.json")
        logger.info(f"| 📁 Tool context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Tool context manager contract path: {self.contract_path}.")

        self._tool_configs: Dict[str, ToolConfig] = {}  # 配置相关参数。
        # 配置相关参数。
        self._tool_history_versions: Dict[str, Dict[str, ToolConfig]] = {}

        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.default_timeout = default_timeout

        self._cleanup_registered = False
        self._faiss_service = None
        self._variables_lock = asyncio.Lock()  # 更新相关状态。

    async def initialize(self, tool_names: Optional[List[str]] = None):
        """初始化组件及其依赖资源。"""

        # 注册相关组件。
        dynamic_manager.register_symbol("TOOL", TOOL)
        dynamic_manager.register_symbol("Tool", Tool)
        dynamic_manager.register_symbol("ToolResponse", ToolResponse)

        # 注册相关组件。
        def tool_context_provider():
            """实现 `tool_context_provider` 的业务逻辑。"""
            return {
                "TOOL": TOOL,
                "Tool": Tool,
                "ToolResponse": ToolResponse,
            }
        dynamic_manager.register_context_provider("tool", tool_context_provider)

        # 初始化相关状态。
        self._faiss_service = FaissService(
            base_dir=self.base_dir,
            model_name=self.model_name
        )

        # 加载所需数据。
        tool_configs = {}
        registry_tool_configs: Dict[str, ToolConfig] = await self._load_from_registry()
        tool_configs.update(registry_tool_configs)

        # 加载所需数据。
        code_tool_configs: Dict[str, ToolConfig] = await self._load_from_code()

        # 配置相关参数。
        for tool_name, code_config in code_tool_configs.items():
            if tool_name in tool_configs:
                registry_config = tool_configs[tool_name]
                # 处理版本与历史记录。
                if version_manager.compare_versions(code_config.version, registry_config.version) > 0:
                    logger.info(f"| 🔄 Overriding tool {tool_name} from registry (v{registry_config.version}) with code version (v{code_config.version})")
                    tool_configs[tool_name] = code_config
                else:
                    logger.info(f"| 📌 Keeping tool {tool_name} from registry (v{registry_config.version}), code version (v{code_config.version}) is not greater")
                    # 配置相关参数。
                    if version_manager.compare_versions(code_config.version, registry_config.version) == 0:
                        # 配置相关参数。
                        if tool_name in self._tool_history_versions:
                            self._tool_history_versions[tool_name][registry_config.version] = registry_config
            else:
                # 处理工具调用。
                tool_configs[tool_name] = code_config

        # 处理工具调用。
        if tool_names is not None:
            tool_configs = {name: tool_configs[name] for name in tool_names}

        # 创建所需对象。
        tool_names = list(tool_configs.keys())
        tasks = [
            self.build(tool_configs[name]) for name in tool_names
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for tool_name, result in zip(tool_names, results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Failed to initialize tool {tool_name}: {result}")
                continue
            self._tool_configs[tool_name] = result
            logger.info(f"| 🔧 Tool {tool_name} initialized")

        # 配置相关参数。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract(tool_names=tool_names)

        # 清理并释放相关资源。
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True

        logger.info(f"| ✅ Tools initialization completed")

    async def _load_from_registry(self):
        """实现 `_load_from_registry` 的业务逻辑。"""

        tool_configs: Dict[str, ToolConfig] = {}

        async def register_tool_class(tool_cls: Type[Tool]):
            """注册与 `register_tool_class` 对应的数据或状态。"""
            try:
                # 配置相关参数。
                tool_config_key = inflection.underscore(tool_cls.__name__)
                tool_config_dict = config.get(tool_config_key, {})
                tool_require_grad = tool_config_dict.get("require_grad", False) if tool_config_dict and "require_grad" in tool_config_dict else False

                # 处理工具调用。
                tool_name = tool_cls.model_fields['name'].default
                tool_description = tool_cls.model_fields['description'].default
                tool_metadata = tool_cls.model_fields['metadata'].default

                # 处理版本与历史记录。
                tool_version = await version_manager.get_version("tool", tool_name)

                # 说明相关实现细节。
                tool_code = dynamic_manager.get_full_module_source(tool_cls)

                tool_parameters = dynamic_manager.get_parameters(tool_cls)
                tool_function_calling = dynamic_manager.build_function_calling(tool_name, tool_description, tool_parameters)
                tool_text = dynamic_manager.build_text_representation(tool_name, tool_description, tool_parameters)
                tool_args_schema = dynamic_manager.build_args_schema(tool_name, tool_parameters)

                # 配置相关参数。
                tool_config = ToolConfig(
                    name=tool_name,
                    description=tool_description,
                    version=tool_version,
                    cls=tool_cls,
                    config=tool_config_dict,
                    instance=None,
                    function_calling=tool_function_calling,
                    text=tool_text,
                    args_schema=tool_args_schema,
                    metadata=tool_metadata,
                    require_grad=tool_require_grad,
                    code=tool_code,
                )

                # 配置相关参数。
                tool_configs[tool_name] = tool_config

                # 处理版本与历史记录。
                if tool_name not in self._tool_history_versions:
                    self._tool_history_versions[tool_name] = {}
                self._tool_history_versions[tool_name][tool_version] = tool_config

                # 注册相关组件。
                await version_manager.register_version("tool", tool_name, tool_version)

                logger.info(f"| 📝 Registered tool: {tool_name} ({tool_cls.__name__})")

            except Exception as e:
                logger.error(f"| ❌ Failed to register tool class {tool_cls.__name__}: {e}")
                raise

        import src.tool  # noqa: F401

        # 注册相关组件。
        tool_classes = list(TOOL._module_dict.values())

        logger.info(f"| 🔍 Discovering {len(tool_classes)} tools from TOOL registry")

        # 注册相关组件。
        tasks = [
            register_tool_class(tool_cls) for tool_cls in tool_classes
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)
        success_count = sum(1 for r in results if not isinstance(r, Exception))

        logger.info(f"| ✅ Discovered and registered {success_count}/{len(tool_classes)} tools from TOOL registry")

        return tool_configs

    async def _load_from_code(self):
        """实现 `_load_from_code` 的业务逻辑。"""

        tool_configs: Dict[str, ToolConfig] = {}

        # 加载所需数据。
        if not os.path.exists(self.save_path):
            logger.info(f"| 📂 Tool config file not found at {self.save_path}, skipping code-based loading")
            return tool_configs

        # 配置相关参数。
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse tool config JSON from {self.save_path}: {e}")
            return tool_configs

        metadata = load_data.get("metadata", {})
        tools_data = load_data.get("tools", {})

        async def register_tool_class(tool_name: str, tool_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, ToolConfig], Optional[ToolConfig]]]:
            """注册与 `register_tool_class` 对应的数据或状态。"""
            try:
                current_version = tool_data.get("current_version", "1.0.0")
                versions = tool_data.get("versions", {})

                if not versions:
                    logger.warning(f"| ⚠️ Tool {tool_name} has no versions")
                    return None

                version_map: Dict[str, ToolConfig] = {}
                current_tool_config: Optional[ToolConfig] = None

                for _, version_data in versions.items():
                    tool_config = ToolConfig.model_validate(version_data)
                    version = tool_config.version
                    version_map[version] = tool_config

                    if version == current_version:
                        current_tool_config = tool_config

                return tool_name, version_map, current_tool_config
            except Exception as e:
                logger.error(f"| ❌ Failed to load tool {tool_name} from code JSON: {e}")
                return None

        # 加载所需数据。
        tasks = [
            register_tool_class(tool_name, tool_data) for tool_name, tool_data in tools_data.items()
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            tool_name, version_map, current_tool_config = result
            if not version_map:
                continue
            # 处理版本与历史记录。
            self._tool_history_versions[tool_name] = version_map
            # 配置相关参数。
            if current_tool_config is not None:
                tool_configs[tool_name] = current_tool_config
            else:
                # 执行回退或重试逻辑。
                logger.warning(f"| ⚠️ Tool {tool_name} current_version not found, using last available version")
                tool_configs[tool_name] = list(version_map.values())[-1]

            # 注册相关组件。
            for tool_config in version_map.values():
                await version_manager.register_version("tool", tool_name, tool_config.version)

        logger.info(f"| 📂 Loaded {len(tool_configs)} tools from {self.save_path}")
        return tool_configs

    async def _store(self, tool_config: ToolConfig):
        """实现 `_store` 的业务逻辑。"""
        if self._faiss_service is None:
            return

        try:
            # 创建所需对象。
            tool_text = f"Tool: {tool_config.name}\nDescription: {tool_config.description}"

            # 说明相关实现细节。
            request = FaissAddRequest(
                texts=[tool_text],
                metadatas=[{
                    "name": tool_config.name,
                    "description": tool_config.description
                }]
            )

            await self._faiss_service.add_documents(request)

        except Exception as e:
            logger.warning(f"| ⚠️ Failed to add tool {tool_config.name} to FAISS index: {e}")

    async def build(self, tool_config: ToolConfig) -> ToolConfig:
        """实现 `build` 的业务逻辑。"""
        if tool_config.name in self._tool_configs:
            existing_config = self._tool_configs[tool_config.name]
            if existing_config.instance is not None:
                return existing_config

        # 创建所需对象。
        try:
            # 加载所需数据。
            if tool_config.cls is None:
                raise ValueError(f"Cannot create tool {tool_config.name}: no class provided. Class should be loaded during initialization.")

            # 处理工具调用。
            tool_instance = tool_config.cls(**tool_config.config) if tool_config.config else tool_config.cls()

            # 初始化相关状态。
            if hasattr(tool_instance, "initialize"):
                await tool_instance.initialize()

            tool_config.instance = tool_instance

            # 处理工具调用。
            self._tool_configs[tool_config.name] = tool_config

            logger.info(f"| 🔧 Tool {tool_config.name} created and stored")

            return tool_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create tool {tool_config.name}: {e}")
            raise

    async def register(self,
                       tool_cls: Type[Tool],
                       tool_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None,
                       code: Optional[str] = None) -> ToolConfig:
        """实现 `register` 的业务逻辑。"""

        try:
            if tool_config_dict is None:
                # 配置相关参数。
                tool_config_key = inflection.underscore(tool_cls.__name__)
                tool_config_dict = config.get(tool_config_key, {})

            # 注册相关组件。
            try:
                tool_instance = tool_cls(**tool_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create tool instance for {tool_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate tool {tool_cls.__name__} with provided config: {e}")

            tool_name = tool_instance.name
            tool_description = tool_instance.description
            tool_metadata = tool_instance.metadata
            # 配置相关参数。
            tool_require_grad = tool_config_dict.get("require_grad", tool_instance.require_grad) if tool_config_dict and "require_grad" in tool_config_dict else tool_instance.require_grad

            # 处理版本与历史记录。
            if version is None:
                tool_version = await version_manager.get_version("tool", tool_name)
            else:
                tool_version = version

            # 处理工具调用。
            tool_code = code if code is not None else dynamic_manager.get_source_code(tool_cls)
            if not tool_code:
                logger.warning(f"| ⚠️ Tool {tool_name} is dynamic but source code cannot be extracted (and no code was provided)")

            # 处理输入参数。
            tool_parameters = dynamic_manager.get_parameters(tool_cls)
            tool_function_calling = dynamic_manager.build_function_calling(tool_name, tool_description, tool_parameters)
            tool_text = dynamic_manager.build_text_representation(tool_name, tool_description, tool_parameters)
            tool_args_schema = dynamic_manager.build_args_schema(tool_name, tool_parameters)

            # 配置相关参数。
            tool_config = ToolConfig(
                name=tool_name,
                description=tool_description,
                metadata=tool_metadata,
                require_grad=tool_require_grad,
                version=tool_version,
                cls=tool_cls,
                config=tool_config_dict or {},
                instance=tool_instance,
                function_calling=tool_function_calling,
                text=tool_text,
                args_schema=tool_args_schema,
                code=tool_code,
            )

            # 配置相关参数。
            self._tool_configs[tool_name] = tool_config

            # 处理版本与历史记录。
            if tool_name not in self._tool_history_versions:
                self._tool_history_versions[tool_name] = {}
            self._tool_history_versions[tool_name][tool_config.version] = tool_config

            # 注册相关组件。
            await version_manager.register_version("tool", tool_name, tool_config.version)

            # 说明相关实现细节。
            await self._store(tool_config)

            # 持久化相关数据。
            await self.save_to_json()
            # 持久化相关数据。
            await self.save_contract()

            logger.info(f"| 📝 Registered tool config: {tool_name}: {tool_config.version}")
            return tool_config

        except Exception as e:
            logger.error(f"| ❌ Failed to register tool: {e}")
            raise


    async def get(self, tool_name: str) -> Tool:
        """实现 `get` 的业务逻辑。"""
        tool_config = self._tool_configs.get(tool_name)
        if tool_config is None:
            return None
        return tool_config.instance if tool_config.instance is not None else None

    async def get_info(self, tool_name: str) -> Optional[ToolConfig]:
        """获取与 `get_info` 对应的数据或状态。"""
        return self._tool_configs.get(tool_name)

    async def list(self) -> List[str]:
        """实现 `list` 的业务逻辑。"""
        return [name for name in self._tool_configs.keys()]

    async def update(self,
                     tool_cls: Type[Tool],
                     tool_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None,
                     description: Optional[str] = None,
                     code: Optional[str] = None) -> ToolConfig:
        """实现 `update` 的业务逻辑。"""
        try:
            if tool_config_dict is None:
                # 配置相关参数。
                tool_config_key = inflection.underscore(tool_cls.__name__)
                tool_config_dict = config.get(tool_config_key, {})

            # 更新相关状态。
            try:
                tool_instance = tool_cls(**tool_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create tool instance for {tool_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate tool {tool_cls.__name__} with provided config: {e}")

            tool_name = tool_instance.name

            # 校验输入与当前状态。
            original_config = self._tool_configs.get(tool_name)
            if original_config is None:
                raise ValueError(f"Tool {tool_name} not found. Use register() to register a new tool.")

            tool_description = tool_instance.description
            tool_metadata = tool_instance.metadata
            # 配置相关参数。
            tool_require_grad = tool_config_dict.get("require_grad", tool_instance.require_grad) if tool_config_dict and "require_grad" in tool_config_dict else tool_instance.require_grad

            # 处理版本与历史记录。
            if new_version is None:
                # 处理版本与历史记录。
                new_version = await version_manager.generate_next_version("tool", tool_name, "patch")

            # 创建所需对象。
            if code is not None:
                tool_code = code
            else:
                tool_code = dynamic_manager.get_source_code(tool_cls)
                if not tool_code:
                    logger.warning(f"| ⚠️ Tool {tool_name} is dynamic but source code cannot be extracted")

            # 创建所需对象。
            tool_parameters = dynamic_manager.get_parameters(tool_cls)
            tool_function_calling = dynamic_manager.build_function_calling(tool_name, tool_description, tool_parameters)
            tool_text = dynamic_manager.build_text_representation(tool_name, tool_description, tool_parameters)
            tool_args_schema = dynamic_manager.build_args_schema(tool_name, tool_parameters)

            # 配置相关参数。
            updated_config = ToolConfig(
                name=tool_name,  # 说明相关实现细节。
                description=tool_description,
                metadata=tool_metadata,
                require_grad=tool_require_grad,
                version=new_version,
                cls=tool_cls,
                config=tool_config_dict or {},
                instance=tool_instance,
                function_calling=tool_function_calling,
                text=tool_text,
                args_schema=tool_args_schema,
                code=tool_code,
            )

            # 配置相关参数。
            self._tool_configs[tool_name] = updated_config

            # 处理版本与历史记录。
            if tool_name not in self._tool_history_versions:
                self._tool_history_versions[tool_name] = {}
            self._tool_history_versions[tool_name][updated_config.version] = updated_config

            # 注册相关组件。
            await version_manager.register_version(
                "tool",
                tool_name,
                new_version,
                description=description or f"Updated from {original_config.version}"
            )

            # 更新相关状态。
            await self._store(updated_config)

            # 持久化相关数据。
            await self.save_to_json()
            # 持久化相关数据。
            await self.save_contract()

            logger.info(f"| 🔄 Updated tool {tool_name} from v{original_config.version} to v{new_version}")
            return updated_config

        except Exception as e:
            logger.error(f"| ❌ Failed to update tool: {e}")
            raise

    async def copy(self,
                  tool_name: str,
                  new_name: Optional[str] = None,
                  new_version: Optional[str] = None,
                  new_config: Optional[Dict[str, Any]] = None) -> ToolConfig:
        """实现 `copy` 的业务逻辑。"""
        try:
            original_config = self._tool_configs.get(tool_name)
            if original_config is None:
                raise ValueError(f"Tool {tool_name} not found")

            if original_config.cls is None:
                raise ValueError(f"Cannot copy tool {tool_name}: no class provided")

            # 说明相关实现细节。
            if new_name is None:
                new_name = tool_name

            # 配置相关参数。
            tool_config_dict = original_config.config.copy() if original_config.config else {}
            if new_config:
                # 配置相关参数。
                tool_config_dict.update(new_config)

            # 处理工具调用。
            try:
                tool_instance = original_config.cls(**tool_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create tool instance for {original_config.cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate tool {original_config.cls.__name__} with provided config: {e}")

            # 说明相关实现细节。
            if new_name != tool_name:
                tool_instance.name = new_name

            tool_description = tool_instance.description
            tool_metadata = tool_instance.metadata
            tool_require_grad = tool_config_dict.get("require_grad", tool_instance.require_grad) if tool_config_dict and "require_grad" in tool_config_dict else tool_instance.require_grad

            # 处理版本与历史记录。
            if new_version is None:
                if new_name == tool_name:
                    # 处理版本与历史记录。
                    new_version = await version_manager.generate_next_version("tool", new_name, "patch")
                else:
                    # 处理版本与历史记录。
                    new_version = await version_manager.get_version("tool", new_name)

            # 处理工具调用。
            tool_code = dynamic_manager.get_source_code(original_config.cls)
            if not tool_code:
                logger.warning(f"| ⚠️ Tool {new_name} is dynamic but source code cannot be extracted")

            # 创建所需对象。
            tool_parameters = dynamic_manager.get_parameters(original_config.cls)
            tool_function_calling = dynamic_manager.build_function_calling(new_name, tool_description, tool_parameters)
            tool_text = dynamic_manager.build_text_representation(new_name, tool_description, tool_parameters)
            tool_args_schema = dynamic_manager.build_args_schema(new_name, tool_parameters)

            # 配置相关参数。
            new_config = ToolConfig(
                name=new_name,
                description=tool_description,
                metadata=tool_metadata,
                require_grad=tool_require_grad,
                version=new_version,
                cls=original_config.cls,
                config=tool_config_dict,
                instance=tool_instance,
                function_calling=tool_function_calling,
                text=tool_text,
                args_schema=tool_args_schema,
                code=tool_code,
            )

            # 注册相关组件。
            self._tool_configs[new_name] = new_config

            # 处理版本与历史记录。
            if new_name not in self._tool_history_versions:
                self._tool_history_versions[new_name] = {}
            self._tool_history_versions[new_name][new_version] = new_config

            # 注册相关组件。
            await version_manager.register_version(
                "tool",
                new_name,
                new_version,
                description=f"Copied from {tool_name}@{original_config.version}"
            )

            # 注册相关组件。
            await self._store(new_config)

            # 持久化相关数据。
            await self.save_to_json()
            # 持久化相关数据。
            await self.save_contract()

            logger.info(f"| 📋 Copied tool {tool_name}@{original_config.version} to {new_name}@{new_version}")
            return new_config

        except Exception as e:
            logger.error(f"| ❌ Failed to copy tool: {e}")
            raise

    async def unregister(self, tool_name: str) -> bool:
        """实现 `unregister` 的业务逻辑。"""
        if tool_name not in self._tool_configs:
            logger.warning(f"| ⚠️ Tool {tool_name} not found")
            return False

        tool_config = self._tool_configs[tool_name]

        # 配置相关参数。
        del self._tool_configs[tool_name]

        # 注册相关组件。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract()

        logger.info(f"| 🗑️ Unregistered tool {tool_name}@{tool_config.version}")
        return True

    async def save_to_json(self, file_path: Optional[str] = None) -> str:
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
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_tools": len(self._tool_configs),
                    "num_versions": sum(len(versions) for versions in self._tool_history_versions.values()),
                },
                "tools": {}
            }

            for tool_name, version_map in self._tool_history_versions.items():
                try:
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for _, tool_config in version_map.items():
                        config_dict = tool_config.model_dump()
                        versions_data[tool_config.version] = config_dict

                    # 配置相关参数。
                    # 配置相关参数。
                    current_version = None
                    if tool_name in self._tool_configs:
                        current_config = self._tool_configs[tool_name]
                        if current_config is not None:
                            current_version = current_config.version

                    # 配置相关参数。
                    if current_version is None and version_map:
                        # 处理版本与历史记录。
                        latest_version_str = None
                        for version_str in version_map.keys():
                            if latest_version_str is None:
                                latest_version_str = version_str
                            elif version_manager.compare_versions(version_str, latest_version_str) > 0:
                                latest_version_str = version_str
                        current_version = latest_version_str

                    save_data["tools"][tool_name] = {
                        "versions": versions_data,
                        "current_version": current_version
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize tool {tool_name}: {e}")
                    continue

            # 持久化相关数据。
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

            logger.info(f"| 💾 Saved {len(self._tool_configs)} tools with version history to {file_path}")
            return str(file_path)

    async def load_from_json(self, file_path: Optional[str] = None, auto_initialize: bool = True) -> bool:
        """加载与 `load_from_json` 对应的数据或状态。"""

        file_path = file_path if file_path is not None else self.save_path

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Tool file not found: {file_path}")
                return False

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)

                tools_data = load_data.get("tools", {})
                loaded_count = 0

                for tool_name, tool_data in tools_data.items():
                    try:
                        # 配置相关参数。
                        versions_data = tool_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(f"| ⚠️ Tool {tool_name} has invalid format for 'versions' (expected dict), skipping")
                            continue

                        current_version_str = tool_data.get("current_version")

                        # 加载所需数据。
                        version_configs = []
                        latest_config = None
                        latest_version = None

                        for version_str, config_dict in versions_data.items():
                            # 处理版本与历史记录。
                            if "version" not in config_dict:
                                config_dict["version"] = version_str

                            try:
                                tool_config = ToolConfig.model_validate(config_dict)
                                version_configs.append(tool_config)
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load tool config for {tool_name}@{version_str}: {e}")
                                continue

                            # 处理版本与历史记录。
                            if latest_config is None or (
                                current_version_str and tool_config.version == current_version_str
                            ) or (
                                not current_version_str and (
                                    latest_version is None or
                                    version_manager.compare_versions(tool_config.version, latest_version) > 0
                                )
                            ):
                                latest_config = tool_config
                                latest_version = tool_config.version

                        # 处理版本与历史记录。
                        self._tool_history_versions[tool_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }

                        # 更新相关状态。
                        if latest_config:
                            self._tool_configs[tool_name] = latest_config

                            # 注册相关组件。
                            for tool_config in version_configs:
                                await version_manager.register_version("tool", tool_name, tool_config.version)

                            # 持久化相关数据。
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)

                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load tool {tool_name}: {e}")
                        continue

                logger.info(f"| 📂 Loaded {loaded_count} tools with version history from {file_path}")
                return True

            except Exception as e:
                logger.error(f"| ❌ Failed to load tools from {file_path}: {e}")
                return False

    async def restore(self, tool_name: str, version: str, auto_initialize: bool = True) -> Optional[ToolConfig]:
        """实现 `restore` 的业务逻辑。"""
        # 处理版本与历史记录。
        version_config = None
        if tool_name in self._tool_history_versions:
            version_config = self._tool_history_versions[tool_name].get(version)

        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for tool {tool_name}")
            return None

        # 创建所需对象。
        restored_config = ToolConfig(**version_config.model_dump())

        # 配置相关参数。
        self._tool_configs[tool_name] = restored_config

        # 更新相关状态。
        version_history = await version_manager.get_version_history("tool", tool_name)
        if version_history:
            # 注册相关组件。
            if version not in version_history.versions:
                await version_manager.register_version("tool", tool_name, version)
            version_history.current_version = version
        else:
            # 注册相关组件。
            await version_manager.register_version("tool", tool_name, version)

        # 初始化相关状态。
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)

        # 持久化相关数据。
        await self.save_to_json()

        logger.info(f"| 🔄 Restored tool {tool_name} to version {version}")
        return restored_config

    async def save_contract(self, tool_names: Optional[List[str]] = None):
        """保存与 `save_contract` 对应的数据或状态。"""
        contract = []
        if tool_names is not None:
            for index, tool_name in enumerate(tool_names):
                tool_info = await self.get_info(tool_name)
                text = tool_info.text
                contract.append(f"{index + 1:04d}\n{text}\n")
        else:
            for index, tool_name in enumerate(self._tool_configs.keys()):
                tool_info = await self.get_info(tool_name)
                text = tool_info.text
                contract.append(f"{index + 1:04d}\n{text}\n")
        contract_text = "---\n".join(contract)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(contract)} tools contract to {self.contract_path}")

    async def load_contract(self) -> str:
        """加载与 `load_contract` 对应的数据或状态。"""
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
        return contract_text

    async def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """实现 `retrieve` 的业务逻辑。"""
        if self._faiss_service is None:
            logger.warning("| ⚠️ FAISS service not initialized, cannot retrieve tools")
            return []

        try:
            from src.environment.faiss.types import FaissSearchRequest

            request = FaissSearchRequest(
                query=query,
                k=k,
                fetch_k=k * 5  # 检索所需信息。
            )

            result = await self._faiss_service.search_similar(request)

            if not result.success:
                logger.warning(f"| ⚠️ FAISS search failed: {result.message}")
                return []

            # 组装并返回结果。
            documents = []
            if result.extra and "documents" in result.extra:
                docs = result.extra["documents"]
                scores = result.extra.get("scores", [])

                for doc, score in zip(docs, scores):
                    # 处理工具调用。
                    metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                    tool_name = metadata.get("name", "")

                    # 配置相关参数。
                    tool_config = None
                    if tool_name and tool_name in self._tool_configs:
                        tool_config = self._tool_configs[tool_name]

                    documents.append({
                        "name": tool_name,
                        "description": metadata.get("description", ""),
                        "score": float(score),
                        "content": doc.get("page_content", "") if isinstance(doc, dict) else str(doc),
                        "config": tool_config.model_dump() if tool_config else None
                    })

            return documents

        except Exception as e:
            logger.error(f"| ❌ Error retrieving tools: {e}")
            return []

    async def cleanup(self):
        """释放组件占用的资源。"""
        try:
            # 配置相关参数。
            self._tool_configs.clear()
            self._tool_history_versions.clear()

            # 执行异步任务。
            if self._faiss_service is not None:
                await self._faiss_service.cleanup()
            logger.info("| 🧹 Tool context manager cleaned up")

        except Exception as e:
            logger.error(f"| ❌ Error during tool context manager cleanup: {e}")

    async def get_variables(self, tool_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_variables` 对应的数据或状态。"""
        # 说明相关实现细节。
        from src.optimizer.types import Variable

        variables: Dict[str, Variable] = {}

        if tool_name is not None:
            # 处理工具调用。
            tool_config = await self.get_info(tool_name)
            if tool_config is None:
                logger.warning(f"| ⚠️ Tool {tool_name} not found")
                return variables

            tool_configs = {tool_name: tool_config}
        else:
            # 处理工具调用。
            tool_configs = self._tool_configs

        for name, tool_config in tool_configs.items():
            # 处理工具调用。
            tool_code = tool_config.code or ""

            # 创建所需对象。
            variable = Variable(
                name=name,
                type="tool_code",
                description=tool_config.description or f"Code for tool {name}",
                require_grad=tool_config.require_grad,
                template=None,
                variables=tool_code  # 说明相关实现细节。
            )
            variables[name] = variable

        return variables

    async def get_trainable_variables(self, tool_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        async with self._variables_lock:
            # 说明相关实现细节。
            all_variables = await self.get_variables(tool_name=tool_name)

            # 说明相关实现细节。
            trainable_variables = {
                name: variable for name, variable in all_variables.items()
                if variable.require_grad is True
            }

            return trainable_variables

    async def set_variables(self, tool_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> ToolConfig:
        """设置与 `set_variables` 对应的数据或状态。"""
        async with self._variables_lock:
            original_config = self._tool_configs.get(tool_name)
            if original_config is None:
                raise ValueError(f"Tool {tool_name} not found. Use register() to register a new tool.")

            # 更新相关状态。
            # 说明相关实现细节。
            if "variables" not in variable_updates:
                raise ValueError(f"variable_updates must contain 'variables' field with tool code, got: {list(variable_updates.keys())}")

            new_code = variable_updates["variables"]
            if not isinstance(new_code, str):
                raise ValueError(f"Tool code must be a string, got {type(new_code)}")

            # 加载所需数据。
            class_name = dynamic_manager.extract_class_name_from_code(new_code)
            if not class_name:
                raise ValueError(f"Cannot extract class name from code")

            try:
                tool_cls = dynamic_manager.load_class(
                    new_code,
                    class_name=class_name,
                    base_class=Tool,
                    context="tool"
                )
            except Exception as e:
                logger.error(f"| ❌ Failed to load tool class from code: {e}")
                raise ValueError(f"Failed to load tool class from code: {e}")

            # 持久化相关数据。
            # 创建所需对象。
            update_description = description or f"Updated code for {tool_name}"
            return await self.update(
                tool_cls=tool_cls,
                tool_config_dict=original_config.config,
                new_version=new_version,
                description=update_description,
                code=new_code  # 创建所需对象。
            )

    async def __call__(self,
                       name: str,
                       input: Dict[str, Any],
                       timeout: Optional[float] = None,
                       ctx: SessionContext = None,
                       **kwargs
                       ) -> ToolResponse:
        """执行组件调用并返回结果。"""

        if ctx is None:
            ctx = SessionContext()

        tool_info = await self.get_info(name)

        if tool_info is None:
            error_msg = f"Tool '{name}' is not registered. Available tools: {list(self._tool_configs.keys())}"
            logger.error(f"| ❌ {error_msg}")
            return ToolResponse(success=False, message=error_msg)

        version = tool_info.version
        tool_instance = tool_info.instance
        logger.info(f"| ✅ Using tool {name}@{version}")

        # 说明相关实现细节。
        effective_timeout = timeout if timeout is not None else self.default_timeout

        # 处理工具调用。
        tool_kwargs = dict(ctx=ctx)

        # 处理工具调用。
        if effective_timeout is None:
            return await tool_instance(**input, **tool_kwargs)

        # 执行异步任务。
        try:
            return await asyncio.wait_for(tool_instance(**input, **tool_kwargs), timeout=effective_timeout)
        except asyncio.TimeoutError:
            error_msg = f"Tool '{name}' execution timed out after {effective_timeout} seconds"
            logger.error(f"| ⏱️ {error_msg}")
            return ToolResponse(
                success=False,
                message=error_msg,
                extra=None
            )
