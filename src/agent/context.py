"""提供上下文管理相关实现。"""
import asyncio
import os
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
from src.utils import (
    assemble_project_path,
    gather_with_concurrency,
    file_lock,
    generate_unique_id
)
from src.agent.types import Agent, AgentConfig
from src.session import SessionContext
from src.version import version_manager
from src.dynamic import dynamic_manager
from src.registry import AGENT


class AgentContextManager(BaseModel):
    """定义 `AgentContextManager`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="The base directory to use for the agents")
    save_path: str = Field(default=None, description="The path to save the agents configuration JSON")
    contract_path: str = Field(default=None, description="The path to save the agent contract")

    def __init__(
        self,
        base_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        contract_path: Optional[str] = None,
        model_name: str = "openrouter/gemini-3-flash-preview",
        embedding_model_name: str = "openrouter/text-embedding-3-large",
        **kwargs: Any,
    ):
        """初始化实例。"""
        super().__init__(**kwargs)

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "agent"))
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| 📁 Agent context manager base directory: {self.base_dir}.")
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "agent.json")
        logger.info(f"| 📁 Agent context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Agent context manager contract path: {self.contract_path}.")

        # 配置相关参数。
        self._agent_configs: Dict[str, AgentConfig] = {}
        # 配置相关参数。
        self._agent_history_versions: Dict[str, Dict[str, AgentConfig]] = {}

        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        self._cleanup_registered = False
        self._faiss_service: Optional[FaissService] = None
        self._variables_lock = asyncio.Lock()  # 更新相关状态。

    async def initialize(self, agent_names: Optional[List[str]] = None) -> None:
        """初始化组件及其依赖资源。"""

        # 注册相关组件。
        dynamic_manager.register_symbol("AGENT", AGENT)
        dynamic_manager.register_symbol("Agent", Agent)
        dynamic_manager.register_symbol("AgentConfig", AgentConfig)

        # 注册相关组件。
        def agent_context_provider():
            return {
                "AGENT": AGENT,
                "Agent": Agent,
                "AgentConfig": AgentConfig,
            }

        dynamic_manager.register_context_provider("agent", agent_context_provider)

        # 初始化相关状态。
        self._faiss_service = FaissService(
            base_dir=self.base_dir,
            model_name=self.model_name,
        )

        # 加载所需数据。
        agent_configs: Dict[str, AgentConfig] = {}
        registry_agent_configs: Dict[str, AgentConfig] = await self._load_from_registry()
        agent_configs.update(registry_agent_configs)

        # 加载所需数据。
        code_agent_configs: Dict[str, AgentConfig] = await self._load_from_code()

        # 配置相关参数。
        for agent_name, code_config in code_agent_configs.items():
            if agent_name in agent_configs:
                registry_config = agent_configs[agent_name]
                if (
                    version_manager.compare_versions(
                        code_config.version, registry_config.version
                    )
                    > 0
                ):
                    logger.info(
                        f"| 🔄 Overriding agent {agent_name} from registry "
                        f"(v{registry_config.version}) with code version (v{code_config.version})"
                    )
                    agent_configs[agent_name] = code_config
                else:
                    logger.info(
                        f"| 📌 Keeping agent {agent_name} from registry (v{registry_config.version}), "
                        f"code version (v{code_config.version}) is not greater"
                    )
                    # 配置相关参数。
                    if version_manager.compare_versions(code_config.version, registry_config.version) == 0:
                        # 配置相关参数。
                        if agent_name in self._agent_history_versions:
                            self._agent_history_versions[agent_name][registry_config.version] = registry_config
            else:
                agent_configs[agent_name] = code_config

        # 说明相关实现细节。
        if agent_names is not None:
            agent_configs = {name: agent_configs[name] for name in agent_names if name in agent_configs}

        # 创建所需对象。
        names = list(agent_configs.keys())
        tasks = [self.build(agent_configs[name]) for name in names]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )

        for agent_name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Failed to initialize agent {agent_name}: {result}")
                continue
            self._agent_configs[agent_name] = result
            logger.info(f"| 🎮 Agent {agent_name} initialized")

        # 配置相关参数。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract(agent_names=agent_names)

        # 清理并释放相关资源。
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True

        logger.info("| ✅ Agents initialization completed")

    async def _load_from_registry(self) -> Dict[str, AgentConfig]:
        """实现 `_load_from_registry` 的业务逻辑。"""

        agent_configs: Dict[str, AgentConfig] = {}

        async def register_agent_class(agent_cls: Type[Agent]):
            """注册与 `register_agent_class` 对应的数据或状态。"""
            try:
                # 配置相关参数。
                agent_config_key = inflection.underscore(agent_cls.__name__)
                agent_config_dict = getattr(config, agent_config_key, {})
                agent_require_grad = agent_config_dict.get("require_grad", False) if agent_config_dict and "require_grad" in agent_config_dict else False

                # 说明相关实现细节。
                agent_name = agent_cls.model_fields['name'].default
                agent_description = agent_cls.model_fields['description'].default
                agent_metadata = agent_cls.model_fields['metadata'].default

                # 处理版本与历史记录。
                agent_version = await version_manager.get_version("agent", agent_name)

                # 说明相关实现细节。
                agent_code = dynamic_manager.get_full_module_source(agent_cls)

                agent_parameters = dynamic_manager.get_parameters(agent_cls)
                agent_function_calling = dynamic_manager.build_function_calling(agent_name, agent_description, agent_parameters)
                agent_text = dynamic_manager.build_text_representation(agent_name, agent_description, agent_parameters)
                agent_args_schema = dynamic_manager.build_args_schema(agent_name, agent_parameters)

                # 配置相关参数。
                agent_config = AgentConfig(
                    name=agent_name,
                    description=agent_description,
                    version=agent_version,
                    require_grad=agent_require_grad,
                    cls=agent_cls,
                    config=agent_config_dict,
                    instance=None,
                    function_calling=agent_function_calling,
                    text=agent_text,
                    args_schema=agent_args_schema,
                    metadata=agent_metadata,
                    code=agent_code,
                )

                # 配置相关参数。
                agent_configs[agent_name] = agent_config

                # 处理版本与历史记录。
                if agent_name not in self._agent_history_versions:
                    self._agent_history_versions[agent_name] = {}
                self._agent_history_versions[agent_name][agent_version] = agent_config

                # 注册相关组件。
                await version_manager.register_version("agent", agent_name, agent_version)

                logger.info(f"| 📝 Registered agent: {agent_name} ({agent_cls.__name__})")

            except Exception as e:
                logger.error(f"| ❌ Failed to register agent class {agent_cls.__name__}: {e}")
                raise

        import src.agent  # noqa: F401

        agent_classes = list(AGENT._module_dict.values())
        logger.info(f"| 🔍 Discovering {len(agent_classes)} agents from AGENT registry")

        tasks = [register_agent_class(agent_cls) for agent_cls in agent_classes]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(
            f"| ✅ Discovered and registered {success_count}/{len(agent_classes)} agents from AGENT registry"
        )

        return agent_configs

    async def _load_from_code(self):
        """实现 `_load_from_code` 的业务逻辑。"""

        agent_configs: Dict[str, AgentConfig] = {}

        # 加载所需数据。
        if not os.path.exists(self.save_path):
            logger.info(f"| 📂 Agent config file not found at {self.save_path}, skipping code-based loading")
            return agent_configs

        # 配置相关参数。
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse agent config JSON from {self.save_path}: {e}")
            return agent_configs

        metadata = load_data.get("metadata", {})
        agents_data = load_data.get("agents", {})

        async def register_agent_class(agent_name: str, agent_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, AgentConfig], Optional[AgentConfig]]]:
            """注册与 `register_agent_class` 对应的数据或状态。"""
            try:
                current_version = agent_data.get("current_version", "1.0.0")
                versions = agent_data.get("versions", {})

                if not versions:
                    logger.warning(f"| ⚠️ Agent {agent_name} has no versions")
                    return None

                version_map: Dict[str, AgentConfig] = {}
                current_agent_config: Optional[AgentConfig] = None

                for _, version_data in versions.items():
                    agent_config = AgentConfig.model_validate(version_data)
                    version = agent_config.version
                    version_map[version] = agent_config

                    if version == current_version:
                        current_agent_config = agent_config

                return agent_name, version_map, current_agent_config
            except Exception as e:
                logger.error(f"| ❌ Failed to load agent {agent_name} from code JSON: {e}")
                return None

        # 加载所需数据。
        tasks = [
            register_agent_class(agent_name, agent_data) for agent_name, agent_data in agents_data.items()
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            agent_name, version_map, current_agent_config = result
            if not version_map:
                continue
            # 处理版本与历史记录。
            self._agent_history_versions[agent_name] = version_map
            # 配置相关参数。
            if current_agent_config is not None:
                agent_configs[agent_name] = current_agent_config
            else:
                # 执行回退或重试逻辑。
                logger.warning(f"| ⚠️ Agent {agent_name} current_version not found, using last available version")
                agent_configs[agent_name] = list(version_map.values())[-1]

            # 注册相关组件。
            for agent_config in version_map.values():
                await version_manager.register_version("agent", agent_name, agent_config.version)

        logger.info(f"| 📂 Loaded {len(agent_configs)} agents from {self.save_path}")
        return agent_configs

    async def _store(self, agent_config: AgentConfig):
        """实现 `_store` 的业务逻辑。"""
        if self._faiss_service is None:
            return

        try:
            # 创建所需对象。
            agent_text = f"Agent: {agent_config.name}\nDescription: {agent_config.description}"

            # 说明相关实现细节。
            request = FaissAddRequest(
                texts=[agent_text],
                metadatas=[{
                    "name": agent_config.name,
                    "description": agent_config.description
                }]
            )

            await self._faiss_service.add_documents(request)

        except Exception as e:
            logger.warning(f"| ⚠️ Failed to add agent {agent_config.name} to FAISS index: {e}")

    async def build(self, agent_config: AgentConfig) -> AgentConfig:
        """实现 `build` 的业务逻辑。"""
        if agent_config.name in self._agent_configs:
            existing_config = self._agent_configs[agent_config.name]
            if existing_config.instance is not None:
                return existing_config

        # 创建所需对象。
        try:
            # 加载所需数据。
            if agent_config.cls is None:
                raise ValueError(f"Cannot create agent {agent_config.name}: no class provided. Class should be loaded during initialization.")

            # 说明相关实现细节。
            agent_instance = agent_config.cls(**agent_config.config) if agent_config.config else agent_config.cls()

            # 初始化相关状态。
            if hasattr(agent_instance, "initialize"):
                await agent_instance.initialize()

            agent_config.instance = agent_instance

            # 说明相关实现细节。
            self._agent_configs[agent_config.name] = agent_config

            logger.info(f"| 🔧 Agent {agent_config.name} created and stored")

            return agent_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create agent {agent_config.name}: {e}")
            raise

    async def register(
        self,
        agent_cls: Type[Agent],
        agent_config_dict: Optional[Dict[str, Any]] = None,
        override: bool = False,
        version: Optional[str] = None,
    ) -> AgentConfig:
        """实现 `register` 的业务逻辑。"""

        try:
            if agent_config_dict is None:
                # 配置相关参数。
                agent_config_key = inflection.underscore(agent_cls.__name__)
                agent_config_dict = getattr(config, agent_config_key, {})

            # 注册相关组件。
            try:
                agent_instance = agent_cls(**agent_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create agent instance for {agent_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate agent {agent_cls.__name__} with provided config: {e}")

            agent_name = agent_instance.name
            agent_description = agent_instance.description
            agent_metadata = agent_instance.metadata
            agent_require_grad = agent_config_dict.get("require_grad", agent_instance.require_grad) if agent_config_dict and "require_grad" in agent_config_dict else agent_instance.require_grad

            # 处理版本与历史记录。
            if version is None:
                agent_version = await version_manager.get_version("agent", agent_name)
            else:
                agent_version = version

            # 说明相关实现细节。
            agent_code = dynamic_manager.get_source_code(agent_cls)
            if not agent_code:
                logger.warning(f"| ⚠️ Agent {agent_name} is dynamic but source code cannot be extracted")

            # 处理输入参数。
            agent_parameters = dynamic_manager.get_parameters(agent_cls)
            agent_function_calling = dynamic_manager.build_function_calling(agent_name, agent_description, agent_parameters)
            agent_text = dynamic_manager.build_text_representation(agent_name, agent_description, agent_parameters)
            agent_args_schema = dynamic_manager.build_args_schema(agent_name, agent_parameters)

            # 配置相关参数。
            agent_config = AgentConfig(
                name=agent_name,
                description=agent_description,
                metadata=agent_metadata,
                version=agent_version,
                require_grad=agent_require_grad,
                cls=agent_cls,
                config=agent_config_dict or {},
                instance=agent_instance,
                function_calling=agent_function_calling,
                text=agent_text,
                args_schema=agent_args_schema,
                code=agent_code,
            )

            # 配置相关参数。
            self._agent_configs[agent_name] = agent_config

            # 处理版本与历史记录。
            if agent_name not in self._agent_history_versions:
                self._agent_history_versions[agent_name] = {}
            self._agent_history_versions[agent_name][agent_config.version] = agent_config

            # 注册相关组件。
            await version_manager.register_version("agent", agent_name, agent_config.version)

            # 说明相关实现细节。
            await self._store(agent_config)

            # 持久化相关数据。
            await self.save_to_json()
            # 持久化相关数据。
            await self.save_contract()

            logger.info(f"| 📝 Registered agent config: {agent_name}: {agent_config.version}")
            return agent_config

        except Exception as e:
            logger.error(f"| ❌ Failed to register agent: {e}")
            raise

    async def get(self, agent_name: str) -> Optional[Agent]:
        """实现 `get` 的业务逻辑。"""
        agent_config = self._agent_configs.get(agent_name)
        if agent_config is None:
            return None
        return agent_config.instance if agent_config.instance is not None else None

    async def get_info(self, agent_name: str) -> Optional[AgentConfig]:
        """获取与 `get_info` 对应的数据或状态。"""
        return self._agent_configs.get(agent_name)

    async def list(self) -> List[str]:
        """实现 `list` 的业务逻辑。"""
        return [name for name in self._agent_configs.keys()]

    async def update(
        self,
        agent_cls: Type[Agent],
        agent_config_dict: Optional[Dict[str, Any]] = None,
        new_version: Optional[str] = None,
        description: Optional[str] = None,
        code: Optional[str] = None,
    ) -> AgentConfig:
        """实现 `update` 的业务逻辑。"""
        try:
            if agent_config_dict is None:
                # 配置相关参数。
                agent_config_key = inflection.underscore(agent_cls.__name__)
                agent_config_dict = getattr(config, agent_config_key, {})

            # 更新相关状态。
            try:
                agent_instance = agent_cls(**agent_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create agent instance for {agent_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate agent {agent_cls.__name__} with provided config: {e}")

            agent_name = agent_instance.name

            # 校验输入与当前状态。
            original_config = self._agent_configs.get(agent_name)
            if original_config is None:
                raise ValueError(f"Agent {agent_name} not found. Use register() to register a new agent.")

            agent_description = agent_instance.description
            agent_metadata = agent_instance.metadata
            agent_require_grad = agent_config_dict.get("require_grad", agent_instance.require_grad) if agent_config_dict else agent_instance.require_grad

            # 处理版本与历史记录。
            if new_version is None:
                # 处理版本与历史记录。
                new_version = await version_manager.generate_next_version("agent", agent_name, "patch")

            # 创建所需对象。
            if code is not None:
                agent_code = code
            else:
                agent_code = dynamic_manager.get_source_code(agent_cls)
                if not agent_code:
                    logger.warning(f"| ⚠️ Agent {agent_name} is dynamic but source code cannot be extracted")

            # 创建所需对象。
            agent_parameters = dynamic_manager.get_parameters(agent_cls)
            agent_function_calling = dynamic_manager.build_function_calling(agent_name, agent_description, agent_parameters)
            agent_text = dynamic_manager.build_text_representation(agent_name, agent_description, agent_parameters)
            agent_args_schema = dynamic_manager.build_args_schema(agent_name, agent_parameters)

            # 配置相关参数。
            updated_config = AgentConfig(
                name=agent_name,  # 说明相关实现细节。
                description=agent_description,
                metadata=agent_metadata,
                version=new_version,
                require_grad=agent_require_grad,
                cls=agent_cls,
                config=agent_config_dict or {},
                instance=agent_instance,
                function_calling=agent_function_calling,
                text=agent_text,
                args_schema=agent_args_schema,
                code=agent_code,
            )

            # 配置相关参数。
            self._agent_configs[agent_name] = updated_config

            # 处理版本与历史记录。
            if agent_name not in self._agent_history_versions:
                self._agent_history_versions[agent_name] = {}
            self._agent_history_versions[agent_name][updated_config.version] = updated_config

            # 注册相关组件。
            await version_manager.register_version(
                "agent",
                agent_name,
                new_version,
                description=description or f"Updated from {original_config.version}"
            )

            # 更新相关状态。
            await self._store(updated_config)

            # 持久化相关数据。
            await self.save_to_json()
            # 持久化相关数据。
            await self.save_contract()

            logger.info(f"| 🔄 Updated agent {agent_name} from v{original_config.version} to v{new_version}")
            return updated_config

        except Exception as e:
            logger.error(f"| ❌ Failed to update agent: {e}")
            raise

    async def copy(
        self,
        agent_name: str,
        new_name: Optional[str] = None,
        new_version: Optional[str] = None,
        new_config: Optional[Dict[str, Any]] = None,
    ) -> AgentConfig:
        """实现 `copy` 的业务逻辑。"""
        try:
            original_config = self._agent_configs.get(agent_name)
            if original_config is None:
                raise ValueError(f"Agent {agent_name} not found")

            if original_config.cls is None:
                raise ValueError(f"Cannot copy agent {agent_name}: no class provided")

            # 说明相关实现细节。
            if new_name is None:
                new_name = agent_name

            # 配置相关参数。
            agent_config_dict = original_config.config.copy() if original_config.config else {}
            if new_config:
                # 配置相关参数。
                agent_config_dict.update(new_config)

            # 说明相关实现细节。
            try:
                agent_instance = original_config.cls(**agent_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create agent instance for {original_config.cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate agent {original_config.cls.__name__} with provided config: {e}")

            # 说明相关实现细节。
            if new_name != agent_name:
                agent_instance.name = new_name

            agent_description = agent_instance.description
            agent_metadata = agent_instance.metadata
            agent_require_grad = agent_config_dict.get("require_grad", agent_instance.require_grad) if agent_config_dict and "require_grad" in agent_config_dict else agent_instance.require_grad

            # 处理版本与历史记录。
            if new_version is None:
                if new_name == agent_name:
                    # 处理版本与历史记录。
                    new_version = await version_manager.generate_next_version("agent", new_name, "patch")
                else:
                    # 处理版本与历史记录。
                    new_version = await version_manager.get_version("agent", new_name)

            # 说明相关实现细节。
            agent_code = dynamic_manager.get_source_code(original_config.cls)
            if not agent_code:
                logger.warning(f"| ⚠️ Agent {new_name} is dynamic but source code cannot be extracted")

            # 创建所需对象。
            agent_parameters = dynamic_manager.get_parameters(original_config.cls)
            agent_function_calling = dynamic_manager.build_function_calling(new_name, agent_description, agent_parameters)
            agent_text = dynamic_manager.build_text_representation(new_name, agent_description, agent_parameters)
            agent_args_schema = dynamic_manager.build_args_schema(new_name, agent_parameters)

            # 配置相关参数。
            new_agent_config = AgentConfig(
                name=new_name,
                description=agent_description,
                metadata=agent_metadata,
                version=new_version,
                require_grad=agent_require_grad,
                cls=original_config.cls,
                config=agent_config_dict,
                instance=agent_instance,
                function_calling=agent_function_calling,
                text=agent_text,
                args_schema=agent_args_schema,
                code=agent_code,
            )

            # 注册相关组件。
            self._agent_configs[new_name] = new_agent_config

            # 处理版本与历史记录。
            if new_name not in self._agent_history_versions:
                self._agent_history_versions[new_name] = {}
            self._agent_history_versions[new_name][new_version] = new_agent_config

            # 注册相关组件。
            await version_manager.register_version(
                "agent",
                new_name,
                new_version,
                description=f"Copied from {agent_name}@{original_config.version}"
            )

            # 注册相关组件。
            await self._store(new_agent_config)

            # 持久化相关数据。
            await self.save_to_json()
            # 持久化相关数据。
            await self.save_contract()

            logger.info(f"| 📋 Copied agent {agent_name}@{original_config.version} to {new_name}@{new_version}")
            return new_agent_config

        except Exception as e:
            logger.error(f"| ❌ Failed to copy agent: {e}")
            raise

    async def unregister(self, agent_name: str) -> bool:
        """实现 `unregister` 的业务逻辑。"""
        if agent_name not in self._agent_configs:
            logger.warning(f"| ⚠️ Agent {agent_name} not found")
            return False

        agent_config = self._agent_configs[agent_name]

        # 配置相关参数。
        del self._agent_configs[agent_name]

        # 注册相关组件。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract()

        logger.info(f"| 🗑️ Unregistered agent {agent_name}@{agent_config.version}")
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
                    "num_agents": len(self._agent_configs),
                    "num_versions": sum(len(versions) for versions in self._agent_history_versions.values()),
                },
                "agents": {}
            }

            for agent_name, version_map in self._agent_history_versions.items():
                try:
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for _, agent_config in version_map.items():
                        config_dict = agent_config.model_dump()
                        versions_data[agent_config.version] = config_dict

                    # 配置相关参数。
                    # 配置相关参数。
                    current_version = None
                    if agent_name in self._agent_configs:
                        current_config = self._agent_configs[agent_name]
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

                    save_data["agents"][agent_name] = {
                        "versions": versions_data,
                        "current_version": current_version,
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize agent {agent_name}: {e}")
                    continue

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

            logger.info(
                f"| 💾 Saved {len(self._agent_configs)} agents with version history to {file_path}"
            )
            return str(file_path)

    async def load_from_json(
        self, file_path: Optional[str] = None, auto_initialize: bool = True
    ) -> bool:
        """加载与 `load_from_json` 对应的数据或状态。"""

        file_path = file_path if file_path is not None else self.save_path

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Agent file not found: {file_path}")
                return False

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)

                agents_data = load_data.get("agents", {})
                loaded_count = 0

                for agent_name, agent_data in agents_data.items():
                    try:
                        # 配置相关参数。
                        versions_data = agent_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(f"| ⚠️ Agent {agent_name} has invalid format for 'versions' (expected dict), skipping")
                            continue

                        current_version_str = agent_data.get("current_version")

                        # 加载所需数据。
                        version_configs = []
                        latest_config = None
                        latest_version = None

                        for version_str, config_dict in versions_data.items():
                            # 处理版本与历史记录。
                            if "version" not in config_dict:
                                config_dict["version"] = version_str

                            try:
                                agent_config = AgentConfig.model_validate(config_dict)
                                version_configs.append(agent_config)
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load agent config for {agent_name}@{version_str}: {e}")
                                continue

                            # 处理版本与历史记录。
                            if latest_config is None or (
                                current_version_str and agent_config.version == current_version_str
                            ) or (
                                not current_version_str and (
                                    latest_version is None or
                                    version_manager.compare_versions(agent_config.version, latest_version) > 0
                                )
                            ):
                                latest_config = agent_config
                                latest_version = agent_config.version

                        # 处理版本与历史记录。
                        self._agent_history_versions[agent_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }

                        # 更新相关状态。
                        if latest_config:
                            self._agent_configs[agent_name] = latest_config

                            # 注册相关组件。
                            for agent_config in version_configs:
                                await version_manager.register_version("agent", agent_name, agent_config.version)

                            # 持久化相关数据。
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)

                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load agent {agent_name}: {e}")
                        continue

                logger.info(f"| 📂 Loaded {loaded_count} agents with version history from {file_path}")
                return True

            except Exception as e:
                logger.error(f"| ❌ Failed to load agents from {file_path}: {e}")
                return False

    async def restore(
        self, agent_name: str, version: str, auto_initialize: bool = True
    ) -> Optional[AgentConfig]:
        """实现 `restore` 的业务逻辑。"""
        # 处理版本与历史记录。
        version_config = None
        if agent_name in self._agent_history_versions:
            version_config = self._agent_history_versions[agent_name].get(version)

        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for agent {agent_name}")
            return None

        # 创建所需对象。
        restored_config = AgentConfig(**version_config.model_dump())

        # 配置相关参数。
        self._agent_configs[agent_name] = restored_config

        # 更新相关状态。
        version_history = await version_manager.get_version_history("agent", agent_name)
        if version_history:
            # 注册相关组件。
            if version not in version_history.versions:
                await version_manager.register_version("agent", agent_name, version)
            version_history.current_version = version
        else:
            # 注册相关组件。
            await version_manager.register_version("agent", agent_name, version)

        # 初始化相关状态。
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)

        # 持久化相关数据。
        await self.save_to_json()

        logger.info(f"| 🔄 Restored agent {agent_name} to version {version}")
        return restored_config

    async def save_contract(self, agent_names: Optional[List[str]] = None):
        """保存与 `save_contract` 对应的数据或状态。"""
        contract = []
        names = agent_names if agent_names is not None else list(self._agent_configs.keys())
        for index, agent_name in enumerate(names):
            agent_info = await self.get_info(agent_name)
            if agent_info is None:
                logger.warning(f"| ⚠️  Skipping agent '{agent_name}' in contract (not found or failed to create)")
                continue
            text = agent_info.text
            contract.append(f"{index + 1:04d}\n{text}\n")
        contract_text = "---\n".join(contract)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(contract)} agents contract to {self.contract_path}")

    async def load_contract(self) -> str:
        """加载与 `load_contract` 对应的数据或状态。"""
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
        return contract_text

    async def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """实现 `retrieve` 的业务逻辑。"""
        if self._faiss_service is None:
            logger.warning("| ⚠️ FAISS service not initialized, cannot retrieve agents")
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
                    # 说明相关实现细节。
                    metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                    agent_name = metadata.get("name", "")

                    # 配置相关参数。
                    agent_config = None
                    if agent_name and agent_name in self._agent_configs:
                        agent_config = self._agent_configs[agent_name]

                    documents.append({
                        "name": agent_name,
                        "description": metadata.get("description", ""),
                        "score": float(score),
                        "content": doc.get("page_content", "") if isinstance(doc, dict) else str(doc),
                        "config": agent_config.model_dump() if agent_config else None
                    })

            return documents

        except Exception as e:
            logger.error(f"| ❌ Error retrieving agents: {e}")
            return []

    async def get_variables(self, agent_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_variables` 对应的数据或状态。"""
        # 说明相关实现细节。
        from src.optimizer.types import Variable

        variables: Dict[str, Variable] = {}

        if agent_name is not None:
            # 说明相关实现细节。
            agent_config = await self.get_info(agent_name)
            if agent_config is None:
                logger.warning(f"| ⚠️ Agent {agent_name} not found")
                return variables

            agent_configs = {agent_name: agent_config}
        else:
            # 说明相关实现细节。
            agent_configs = self._agent_configs

        for name, agent_config in agent_configs.items():
            # 说明相关实现细节。
            agent_code = ""
            if agent_config.cls is not None:
                agent_code = dynamic_manager.get_full_module_source(agent_config.cls) or ""
            elif agent_config.code:
                agent_code = agent_config.code

            # 创建所需对象。
            variable = Variable(
                name=name,
                type="agent_code",
                description=agent_config.description or f"Code for agent {name}",
                require_grad=agent_config.require_grad,
                template=None,
                variables=agent_code  # 说明相关实现细节。
            )
            variables[name] = variable

        return variables

    async def get_trainable_variables(self, agent_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        async with self._variables_lock:
            all_variables = await self.get_variables(agent_name=agent_name)
            trainable_variables = {name: var for name, var in all_variables.items() if var.require_grad}
            return trainable_variables

    async def set_variables(self, agent_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> AgentConfig:
        """设置与 `set_variables` 对应的数据或状态。"""
        async with self._variables_lock:
            original_config = self._agent_configs.get(agent_name)
            if original_config is None:
                raise ValueError(f"Agent {agent_name} not found. Use register() to register a new agent.")

            # 更新相关状态。
            # 说明相关实现细节。
            if "variables" not in variable_updates:
                raise ValueError(f"variable_updates must contain 'variables' field with agent code, got: {list(variable_updates.keys())}")

            new_code = variable_updates["variables"]
            if not isinstance(new_code, str):
                raise ValueError(f"Agent code must be a string, got {type(new_code)}")

            # 加载所需数据。
            class_name = dynamic_manager.extract_class_name_from_code(new_code)
            if not class_name:
                raise ValueError(f"Cannot extract class name from code")

            try:
                agent_cls = dynamic_manager.load_class(
                    new_code,
                    class_name=class_name,
                    base_class=Agent,
                    context="agent"
                )
            except Exception as e:
                logger.error(f"| ❌ Failed to load agent class from code: {e}")
                raise ValueError(f"Failed to load agent class from code: {e}")

            # 持久化相关数据。
            # 创建所需对象。
            update_description = description or f"Updated code for {agent_name}"
            return await self.update(
                agent_cls=agent_cls,
                agent_config_dict=original_config.config,
                new_version=new_version,
                description=update_description,
                code=new_code  # 创建所需对象。
            )

    async def cleanup(self):
        """释放组件占用的资源。"""
        try:
            # 配置相关参数。
            self._agent_configs.clear()
            self._agent_history_versions.clear()

            # 执行异步任务。
            if self._faiss_service is not None:
                await self._faiss_service.cleanup()
            logger.info("| 🧹 Agent context manager cleaned up")

        except Exception as e:
            logger.error(f"| ❌ Error during agent context manager cleanup: {e}")

    async def __call__(self, name: str, input: Dict[str, Any], ctx: SessionContext = None, **kwargs) -> Any:
        """执行组件调用并返回结果。"""
        if ctx is None:
            ctx = SessionContext()

        agent_info = await self.get_info(name)

        # 说明相关实现细节。
        agent_args = {
            "ctx": ctx,
            **kwargs,
        }

        version = agent_info.version
        agent_instance = agent_info.instance
        logger.info(f"| ✅ Using agent {name}@{version}")

        return await agent_instance(**input, **agent_args)
