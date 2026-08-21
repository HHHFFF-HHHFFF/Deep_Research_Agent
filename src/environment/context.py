"""提供上下文管理相关实现。"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
)

import inflection
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.optimizer.types import Variable
import builtins

from asyncio_atexit import register as async_atexit_register

from src.config import config
from src.dynamic import dynamic_manager
from src.environment.faiss.service import FaissService
from src.environment.faiss.types import FaissAddRequest
from src.environment.types import ActionConfig, Environment, EnvironmentConfig
from src.logger import logger
from src.registry import ENVIRONMENT, load_builtin_components
from src.session import SessionContext
from src.utils import (
    assemble_project_path,
    gather_with_concurrency,
    read_json_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from src.utils.file_utils import file_lock
from src.version import version_manager


class EnvironmentContextManager(BaseModel):
    """定义 `EnvironmentContextManager`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(
        default=None, description="The base directory to use for the environments"
    )
    save_path: str = Field(
        default=None, description="The path to save the environments"
    )
    contract_path: str = Field(
        default=None, description="The path to save the environment contract"
    )

    def __init__(
        self,
        base_dir: str | None = None,
        save_path: str | None = None,
        contract_path: str | None = None,
        model_name: str = "openrouter/gemini-3-flash-preview",
        embedding_model_name: str = "openrouter/text-embedding-3-large",
        **kwargs,
    ):
        """初始化实例。"""
        super().__init__(**kwargs)

        # 更新相关状态。
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(
                os.path.join(config.workdir, "environment")
            )
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(
            f"| 📁 Environment context manager base directory: {self.base_dir}."
        )
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "environment.json")
        logger.info(f"| 📁 Environment context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(
            f"| 📁 Environment context manager contract path: {self.contract_path}."
        )

        self._environment_configs: dict[str, EnvironmentConfig] = {}  # 配置相关参数。
        # 配置相关参数。
        self._environment_history_versions: dict[str, dict[str, EnvironmentConfig]] = {}

        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        self._cleanup_registered = False
        self._faiss_service = None
        self._variables_lock = asyncio.Lock()  # 更新相关状态。

    async def initialize(self, env_names: list[str] | None = None):
        """初始化组件及其依赖资源。"""

        # 注册相关组件。
        dynamic_manager.register_symbol("ENVIRONMENT", ENVIRONMENT)
        dynamic_manager.register_symbol("Environment", Environment)
        dynamic_manager.register_symbol("EnvironmentConfig", EnvironmentConfig)
        dynamic_manager.register_symbol("ActionConfig", ActionConfig)

        # 注册相关组件。
        def environment_context_provider():
            """实现 `environment_context_provider` 的业务逻辑。"""
            return {
                "ENVIRONMENT": ENVIRONMENT,
                "Environment": Environment,
                "EnvironmentConfig": EnvironmentConfig,
                "ActionConfig": ActionConfig,
            }

        dynamic_manager.register_context_provider(
            "environment", environment_context_provider
        )

        # 初始化相关状态。
        self._faiss_service = FaissService(
            base_dir=self.base_dir, model_name=self.model_name
        )

        # 加载所需数据。
        env_configs = {}
        registry_env_configs: dict[
            str, EnvironmentConfig
        ] = await self._load_from_registry()
        env_configs.update(registry_env_configs)

        # 加载所需数据。
        code_configs: dict[str, EnvironmentConfig] = await self._load_from_code()

        # 配置相关参数。
        for env_name, code_config in code_configs.items():
            if env_name in env_configs:
                registry_config = env_configs[env_name]
                # 处理版本与历史记录。
                if (
                    version_manager.compare_versions(
                        code_config.version, registry_config.version
                    )
                    > 0
                ):
                    logger.info(
                        f"| 🔄 Overriding environment {env_name} from registry (v{registry_config.version}) with code version (v{code_config.version})"
                    )
                    env_configs[env_name] = code_config
                else:
                    logger.info(
                        f"| 📌 Keeping environment {env_name} from registry (v{registry_config.version}), code version (v{code_config.version}) is not greater"
                    )
                    # 配置相关参数。
                    if (
                        version_manager.compare_versions(
                            code_config.version, registry_config.version
                        )
                        == 0
                        and env_name in self._environment_history_versions
                    ):
                        self._environment_history_versions[env_name][
                            registry_config.version
                        ] = registry_config
            else:
                # 说明相关实现细节。
                env_configs[env_name] = code_config

        # 说明相关实现细节。
        if env_names is not None:
            env_configs = {
                name: env_configs[name] for name in env_names if name in env_configs
            }

        # 创建所需对象。
        env_names_list = list(env_configs.keys())
        tasks = [self.build(env_configs[name]) for name in env_names_list]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )

        for env_name, result in zip(env_names_list, results):
            if isinstance(result, Exception):
                logger.error(
                    f"| ❌ Failed to initialize environment {env_name}: {result}"
                )
                continue
            self._environment_configs[env_name] = result
            logger.info(f"| 🎮 Environment {env_name} initialized")

        # 配置相关参数。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract(env_names=env_names)

        # 清理并释放相关资源。
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True

        logger.info("| ✅ Environments initialization completed")

    async def _load_from_registry(self):
        """实现 `_load_from_registry` 的业务逻辑。"""

        env_configs: dict[str, EnvironmentConfig] = {}

        async def register_environment_class(env_cls: type[Environment]):
            """注册与 `register_environment_class` 对应的数据或状态。"""
            try:
                env_config_key = inflection.underscore(env_cls.__name__)
                env_config_dict = config.get(env_config_key, {})
                env_require_grad = (
                    env_config_dict.get("require_grad", False)
                    if env_config_dict and "require_grad" in env_config_dict
                    else False
                )

                # 说明相关实现细节。
                env_name = env_cls.model_fields["name"].default
                env_description = env_cls.model_fields["description"].default
                env_metadata = env_cls.model_fields["metadata"].default

                # 处理版本与历史记录。
                env_version = await version_manager.get_version("environment", env_name)

                # 说明相关实现细节。
                env_code = dynamic_manager.get_full_module_source(env_cls)

                # 创建所需对象。
                env_actions = {}
                for attr_name in dir(env_cls):
                    attr = getattr(env_cls, attr_name)
                    if hasattr(attr, "_action_name"):
                        action_name = attr._action_name
                        action_description = getattr(attr, "_action_description", "")
                        action_function = getattr(attr, "_action_function", None)
                        action_metadata = getattr(attr, "_action_metadata", {})

                        action_version = await version_manager.get_version(
                            "action", action_name
                        )

                        action_code = dynamic_manager.get_source_code(attr)
                        if not action_code:
                            logger.warning(
                                f"| ⚠️ Action {action_name} is dynamic but source code cannot be extracted"
                            )

                        action_parameters = dynamic_manager.get_parameters(
                            action_function
                        )
                        action_function_calling = (
                            dynamic_manager.build_function_calling(
                                action_name, action_description, action_parameters
                            )
                        )
                        action_text = dynamic_manager.build_text_representation(
                            action_name, action_description, action_parameters
                        )
                        action_args_schema = dynamic_manager.build_args_schema(
                            action_name, action_parameters
                        )

                        action_config = ActionConfig(
                            env_name=env_name,
                            name=action_name,
                            description=action_description,
                            function=action_function,
                            metadata=action_metadata,
                            version=action_version,
                            code=action_code,
                            function_calling=action_function_calling,
                            text=action_text,
                            args_schema=action_args_schema,
                        )

                        env_actions[action_name] = action_config

                # 配置相关参数。
                env_config = EnvironmentConfig(
                    name=env_name,
                    description=env_description,
                    metadata=env_metadata,
                    version=env_version,
                    require_grad=env_require_grad,
                    cls=env_cls,
                    config=env_config_dict,
                    instance=None,
                    code=env_code,
                    actions=env_actions,
                    rules="",  # 说明相关实现细节。
                )

                env_configs[env_name] = env_config

                # 处理版本与历史记录。
                if env_name not in self._environment_history_versions:
                    self._environment_history_versions[env_name] = {}
                self._environment_history_versions[env_name][env_version] = env_config

                # 注册相关组件。
                await version_manager.register_version(
                    "environment", env_name, env_version
                )

                logger.info(
                    f"| 📝 Registered environment: {env_name} ({env_cls.__name__})"
                )

            except Exception as e:
                logger.error(
                    f"| ❌ Failed to register environment class {env_cls.__name__}: {e}"
                )
                raise

        load_builtin_components("environment")

        # 注册相关组件。
        environment_classes = list(ENVIRONMENT._module_dict.values())

        logger.info(
            f"| 🔍 Discovering {len(environment_classes)} environments from ENVIRONMENT registry"
        )

        # 注册相关组件。
        tasks = [register_environment_class(env_cls) for env_cls in environment_classes]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )
        success_count = sum(
            1 for r in results if r is not None and not isinstance(r, Exception)
        )

        logger.info(
            f"| ✅ Discovered and registered {success_count}/{len(environment_classes)} environments from ENVIRONMENT registry"
        )

        return env_configs

    async def _load_from_code(self):
        """实现 `_load_from_code` 的业务逻辑。"""

        env_configs: dict[str, EnvironmentConfig] = {}

        # 加载所需数据。
        if not os.path.exists(self.save_path):
            logger.info(
                f"| 📂 Environment config file not found at {self.save_path}, skipping code-based loading"
            )
            return env_configs

        # 配置相关参数。
        try:
            load_data = await read_json_file(self.save_path)
        except json.JSONDecodeError as e:
            logger.warning(
                f"| ⚠️ Failed to parse environment config JSON from {self.save_path}: {e}"
            )
            return env_configs

        environments_data = load_data.get("environments", {})

        async def register_environment_class(
            env_name: str, env_data: dict[str, Any]
        ) -> tuple[str, dict[str, EnvironmentConfig], EnvironmentConfig | None] | None:
            """注册与 `register_environment_class` 对应的数据或状态。"""
            try:
                current_version = env_data.get("current_version", "1.0.0")
                versions = env_data.get("versions", {})

                if not versions:
                    logger.warning(f"| ⚠️ Environment {env_name} has no versions")
                    return None

                version_map: dict[str, EnvironmentConfig] = {}
                current_config: EnvironmentConfig | None = None  # 配置相关参数。

                for version_data in versions.values():
                    env_config = EnvironmentConfig.from_dict(version_data)
                    version = env_config.version
                    version_map[version] = env_config

                    if version == current_version:
                        current_config = env_config

                return env_name, version_map, current_config
            except Exception as e:
                logger.error(
                    f"| ❌ Failed to load environment {env_name} from code JSON: {e}"
                )
                return None

        # 加载所需数据。
        tasks = [
            register_environment_class(env_name, env_data)
            for env_name, env_data in environments_data.items()
        ]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            env_name, version_map, current_environment_config = result
            if not version_map:
                continue
            # 处理版本与历史记录。
            self._environment_history_versions[env_name] = version_map
            # 配置相关参数。
            if current_environment_config is not None:
                env_configs[env_name] = current_environment_config
            else:
                # 执行回退或重试逻辑。
                logger.warning(
                    f"| ⚠️ Environment {env_name} current_version not found, using last available version"
                )
                env_configs[env_name] = list(version_map.values())[-1]

            # 注册相关组件。
            for env_config in version_map.values():
                await version_manager.register_version(
                    "environment", env_name, env_config.version
                )

        logger.info(
            f"| 📂 Loaded {len(env_configs)} environments from {self.save_path}"
        )
        return env_configs

    async def _store(self, env_config: EnvironmentConfig):
        """实现 `_store` 的业务逻辑。"""
        if self._faiss_service is None:
            return

        try:
            # 创建所需对象。
            env_text = (
                f"Environment: {env_config.name}\nDescription: {env_config.description}"
            )

            # 处理工具调用。
            if env_config.actions:
                action_descriptions = [
                    f"{name}: {action.description}"
                    for name, action in env_config.actions.items()
                ]
                if action_descriptions:
                    env_text += f"\nActions: {'; '.join(action_descriptions)}"

            # 说明相关实现细节。
            request = FaissAddRequest(
                texts=[env_text],
                metadatas=[
                    {
                        "name": env_config.name,
                        "description": env_config.description,
                        "version": env_config.version,
                    }
                ],
            )

            await self._faiss_service.add_documents(request)

        except Exception as e:
            logger.warning(
                f"| ⚠️ Failed to add environment {env_config.name} to FAISS index: {e}"
            )

    async def build(self, env_config: EnvironmentConfig) -> EnvironmentConfig:
        """实现 `build` 的业务逻辑。"""
        if env_config.name in self._environment_configs:
            existing_config = self._environment_configs[env_config.name]
            if existing_config.instance is not None:
                return existing_config

        try:
            if env_config.cls is None:
                raise ValueError(
                    f"Cannot create environment {env_config.name}: no class provided. Class should be loaded during initialization."
                )

            env_instance = (
                env_config.cls(**env_config.config)
                if env_config.config
                else env_config.cls()
            )

            # 初始化相关状态。
            if hasattr(env_instance, "initialize"):
                await env_instance.initialize()

            env_config.instance = env_instance

            # 加载所需数据。
            if not env_config.rules:
                env_config.rules = env_instance.get_rules()

            # 说明相关实现细节。
            self._environment_configs[env_config.name] = env_config

            logger.info(f"| ✅ Environment {env_config.name} created and stored")

            return env_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create environment {env_config.name}: {e}")
            raise

    async def register(
        self,
        env_cls: type[Environment],
        env_config_dict: dict[str, Any] | None = None,
        override: bool = False,
        version: str | None = None,
    ) -> EnvironmentConfig:
        """实现 `register` 的业务逻辑。"""
        try:
            if env_config_dict is None:
                # 配置相关参数。
                env_config_key = inflection.underscore(env_cls.__name__)
                env_config_dict = (
                    getattr(config, env_config_key, {})
                    if hasattr(config, env_config_key)
                    else {}
                )

            # 注册相关组件。
            try:
                env_instance = env_cls(**env_config_dict)
            except Exception as e:
                logger.error(
                    f"| ❌ Failed to create environment instance for {env_cls.__name__}: {e}"
                )
                raise ValueError(
                    f"Failed to instantiate environment {env_cls.__name__} with provided config: {e}"
                )

            env_name = env_instance.name
            env_description = env_instance.description
            env_metadata = getattr(env_instance, "metadata", {})
            env_require_grad = getattr(env_instance, "require_grad", False)

            if not env_name:
                raise ValueError("Environment.name cannot be empty.")

            if env_name in self._environment_configs and not override:
                raise ValueError(
                    f"Environment '{env_name}' already registered. Use override=True to replace it."
                )

            # 处理版本与历史记录。
            if version is None:
                env_version = await version_manager.get_version("environment", env_name)
            else:
                env_version = version

            # 说明相关实现细节。
            env_code = dynamic_manager.get_full_module_source(env_cls)

            # 加载所需数据。
            actions = {}
            for attr_name in dir(env_cls):
                attr = getattr(env_cls, attr_name)
                if hasattr(attr, "_action_name"):
                    action_name = attr._action_name
                    action_description = getattr(attr, "_action_description", "")
                    action_function = getattr(attr, "_action_function", None)
                    action_metadata = getattr(attr, "_action_metadata", {})

                    action_version = await version_manager.get_version(
                        "action", action_name
                    )

                    action_code = dynamic_manager.get_source_code(attr)
                    if not action_code:
                        logger.warning(
                            f"| ⚠️ Action {action_name} is dynamic but source code cannot be extracted"
                        )

                    action_parameters = dynamic_manager.get_parameters(action_function)
                    action_function_calling = dynamic_manager.build_function_calling(
                        action_name, action_description, action_parameters
                    )
                    action_text = dynamic_manager.build_text_representation(
                        action_name, action_description, action_parameters
                    )
                    action_args_schema = dynamic_manager.build_args_schema(
                        action_name, action_parameters
                    )

                    action_config = ActionConfig(
                        env_name=env_name,
                        name=action_name,
                        description=action_description,
                        function=action_function,
                        metadata=action_metadata,
                        version=action_version,
                        code=action_code,
                        function_calling=action_function_calling,
                        text=action_text,
                        args_schema=action_args_schema,
                    )

                    actions[action_name] = action_config

            # 说明相关实现细节。
            env_rules = (
                env_instance.get_rules() if hasattr(env_instance, "get_rules") else ""
            )

            # 配置相关参数。
            env_config = EnvironmentConfig(
                name=env_name,
                description=env_description,
                rules=env_rules,
                version=env_version,
                require_grad=env_require_grad,
                actions=actions,
                cls=env_cls,
                config=env_config_dict or {},
                instance=env_instance,
                metadata=env_metadata,
                code=env_code,
            )

            # 配置相关参数。
            self._environment_configs[env_name] = env_config

            # 处理版本与历史记录。
            if env_name not in self._environment_history_versions:
                self._environment_history_versions[env_name] = {}
            self._environment_history_versions[env_name][env_config.version] = (
                env_config
            )

            # 注册相关组件。
            await version_manager.register_version(
                "environment", env_name, env_config.version
            )

            # 说明相关实现细节。
            await self._store(env_config)

            # 持久化相关数据。
            await self.save_to_json()

            logger.info(
                f"| 📝 Registered environment config: {env_name}: {env_config.version}"
            )
            return env_config

        except Exception as e:
            logger.error(f"| ❌ Failed to register environment: {e}")
            raise

    async def get(self, env_name: str) -> Environment | None:
        """实现 `get` 的业务逻辑。"""
        env_config = self._environment_configs.get(env_name)
        if env_config:
            return env_config.instance
        return None

    async def get_info(self, env_name: str) -> EnvironmentConfig | None:
        """获取与 `get_info` 对应的数据或状态。"""
        return self._environment_configs.get(env_name)

    async def get_state(
        self, env_name: str, ctx: SessionContext = None, **kwargs
    ) -> dict[str, Any] | None:
        """获取与 `get_state` 对应的数据或状态。"""

        if ctx is None:
            ctx = SessionContext()

        env_args = {
            "ctx": ctx,
        }

        env_config = self._environment_configs.get(env_name)
        if not env_config or not env_config.instance:
            raise ValueError(f"Environment '{env_name}' not found")
        return await env_config.instance.get_state(**env_args)

    async def list(self) -> list[str]:
        """实现 `list` 的业务逻辑。"""
        return [name for name in self._environment_configs]

    async def update(
        self,
        env_cls: type[Environment],
        env_config_dict: dict[str, Any] | None = None,
        new_version: str | None = None,
        description: str | None = None,
        code: str | None = None,
    ) -> EnvironmentConfig:
        """实现 `update` 的业务逻辑。"""
        try:
            if env_config_dict is None:
                # 配置相关参数。
                env_config_key = inflection.underscore(env_cls.__name__)
                env_config_dict = (
                    getattr(config, env_config_key, {})
                    if hasattr(config, env_config_key)
                    else {}
                )

            # 更新相关状态。
            try:
                env_instance = env_cls(**env_config_dict)
            except Exception as e:
                logger.error(
                    f"| ❌ Failed to create environment instance for {env_cls.__name__}: {e}"
                )
                raise ValueError(
                    f"Failed to instantiate environment {env_cls.__name__} with provided config: {e}"
                )

            env_name = env_instance.name

            # 校验输入与当前状态。
            original_config = self._environment_configs.get(env_name)
            if original_config is None:
                raise ValueError(
                    f"Environment {env_name} not found. Use register() to register a new environment."
                )

            env_description = env_instance.description
            env_metadata = getattr(env_instance, "metadata", {})
            env_require_grad = (
                env_config_dict.get(
                    "require_grad", getattr(env_instance, "require_grad", False)
                )
                if env_config_dict and "require_grad" in env_config_dict
                else getattr(env_instance, "require_grad", False)
            )

            # 处理版本与历史记录。
            if new_version is None:
                # 处理版本与历史记录。
                new_version = await version_manager.generate_next_version(
                    "environment", env_name, "patch"
                )

            # 创建所需对象。
            if code is not None:
                env_code = code
            else:
                env_code = dynamic_manager.get_full_module_source(env_cls)

            # 注册相关组件。
            actions = {}
            for attr_name in dir(env_cls):
                attr = getattr(env_cls, attr_name)
                if hasattr(attr, "_action_name"):
                    action_name = attr._action_name
                    action_description = getattr(attr, "_action_description", "")
                    action_function = getattr(attr, "_action_function", None)
                    action_metadata = getattr(attr, "_action_metadata", {})

                    action_version = await version_manager.get_version(
                        "action", action_name
                    )

                    action_code = dynamic_manager.get_source_code(attr)
                    if not action_code:
                        logger.warning(
                            f"| ⚠️ Action {action_name} is dynamic but source code cannot be extracted"
                        )

                    action_parameters = dynamic_manager.get_parameters(action_function)
                    action_function_calling = dynamic_manager.build_function_calling(
                        action_name, action_description, action_parameters
                    )
                    action_text = dynamic_manager.build_text_representation(
                        action_name, action_description, action_parameters
                    )
                    action_args_schema = dynamic_manager.build_args_schema(
                        action_name, action_parameters
                    )

                    action_config = ActionConfig(
                        env_name=env_name,
                        name=action_name,
                        description=action_description,
                        function=action_function,
                        metadata=action_metadata,
                        version=action_version,
                        code=action_code,
                        function_calling=action_function_calling,
                        text=action_text,
                        args_schema=action_args_schema,
                    )

                    actions[action_name] = action_config

            # 说明相关实现细节。
            env_rules = (
                env_instance.get_rules() if hasattr(env_instance, "get_rules") else ""
            )

            # 配置相关参数。
            updated_config = EnvironmentConfig(
                name=env_name,  # 说明相关实现细节。
                description=env_description,
                rules=env_rules,
                version=new_version,
                require_grad=env_require_grad,
                actions=actions,
                cls=env_cls,
                config=env_config_dict or {},
                instance=env_instance,
                metadata=env_metadata,
                code=env_code,
            )

            # 配置相关参数。
            self._environment_configs[env_name] = updated_config

            # 处理版本与历史记录。
            if env_name not in self._environment_history_versions:
                self._environment_history_versions[env_name] = {}
            self._environment_history_versions[env_name][updated_config.version] = (
                updated_config
            )

            # 注册相关组件。
            await version_manager.register_version(
                "environment",
                env_name,
                new_version,
                description=description or f"Updated from {original_config.version}",
            )

            # 更新相关状态。
            await self._store(updated_config)

            # 持久化相关数据。
            await self.save_to_json()

            logger.info(
                f"| 🔄 Updated environment {env_name} from v{original_config.version} to v{new_version}"
            )
            return updated_config

        except Exception as e:
            logger.error(f"| ❌ Failed to update environment: {e}")
            raise

    async def copy(
        self,
        env_name: str,
        new_name: str | None = None,
        new_version: str | None = None,
        new_config: dict[str, Any] | None = None,
    ) -> EnvironmentConfig:
        """实现 `copy` 的业务逻辑。"""
        try:
            original_config = self._environment_configs.get(env_name)
            if original_config is None:
                raise ValueError(f"Environment {env_name} not found")

            if original_config.cls is None:
                raise ValueError(
                    f"Cannot copy environment {env_name}: no class provided"
                )

            # 说明相关实现细节。
            if new_name is None:
                new_name = env_name

            # 配置相关参数。
            env_config_dict = (
                original_config.config.copy() if original_config.config else {}
            )
            if new_config:
                # 配置相关参数。
                env_config_dict.update(new_config)

            # 说明相关实现细节。
            try:
                env_instance = original_config.cls(**env_config_dict)
            except Exception as e:
                logger.error(
                    f"| ❌ Failed to create environment instance for {original_config.cls.__name__}: {e}"
                )
                raise ValueError(
                    f"Failed to instantiate environment {original_config.cls.__name__} with provided config: {e}"
                )

            # 说明相关实现细节。
            if new_name != env_name:
                env_instance.name = new_name

            env_description = env_instance.description
            env_metadata = getattr(env_instance, "metadata", {})
            env_require_grad = (
                env_config_dict.get(
                    "require_grad", getattr(env_instance, "require_grad", False)
                )
                if env_config_dict and "require_grad" in env_config_dict
                else getattr(env_instance, "require_grad", False)
            )

            # 处理版本与历史记录。
            if new_version is None:
                if new_name == env_name:
                    # 处理版本与历史记录。
                    new_version = await version_manager.generate_next_version(
                        "environment", new_name, "patch"
                    )
                else:
                    # 处理版本与历史记录。
                    new_version = await version_manager.get_version(
                        "environment", new_name
                    )

            # 说明相关实现细节。
            env_code = dynamic_manager.get_full_module_source(original_config.cls)

            # 注册相关组件。
            actions = {}
            for attr_name in dir(original_config.cls):
                attr = getattr(original_config.cls, attr_name)
                if hasattr(attr, "_action_name"):
                    action_name = attr._action_name
                    action_description = getattr(attr, "_action_description", "")
                    action_function = getattr(attr, "_action_function", None)
                    action_metadata = getattr(attr, "_action_metadata", {})

                    action_version = await version_manager.get_version(
                        "action", action_name
                    )

                    action_code = dynamic_manager.get_source_code(attr)
                    if not action_code:
                        logger.warning(
                            f"| ⚠️ Action {action_name} is dynamic but source code cannot be extracted"
                        )

                    action_parameters = dynamic_manager.get_parameters(action_function)
                    action_function_calling = dynamic_manager.build_function_calling(
                        action_name, action_description, action_parameters
                    )
                    action_text = dynamic_manager.build_text_representation(
                        action_name, action_description, action_parameters
                    )
                    action_args_schema = dynamic_manager.build_args_schema(
                        action_name, action_parameters
                    )

                    action_config = ActionConfig(
                        env_name=new_name,
                        name=action_name,
                        description=action_description,
                        function=action_function,
                        metadata=action_metadata,
                        version=action_version,
                        code=action_code,
                        function_calling=action_function_calling,
                        text=action_text,
                        args_schema=action_args_schema,
                    )

                    actions[action_name] = action_config

            # 说明相关实现细节。
            env_rules = (
                env_instance.get_rules() if hasattr(env_instance, "get_rules") else ""
            )

            # 配置相关参数。
            copied_config = EnvironmentConfig(
                name=new_name,
                description=env_description,
                rules=env_rules,
                version=new_version,
                require_grad=env_require_grad,
                actions=actions,
                cls=original_config.cls,
                config=env_config_dict,
                instance=env_instance,
                metadata=env_metadata,
                code=env_code,
            )

            # 注册相关组件。
            self._environment_configs[new_name] = copied_config

            # 处理版本与历史记录。
            if new_name not in self._environment_history_versions:
                self._environment_history_versions[new_name] = {}
            self._environment_history_versions[new_name][new_version] = copied_config

            # 注册相关组件。
            await version_manager.register_version(
                "environment",
                new_name,
                new_version,
                description=f"Copied from {env_name}@{original_config.version}",
            )

            # 注册相关组件。
            await self._store(copied_config)

            # 持久化相关数据。
            await self.save_to_json()

            logger.info(
                f"| 📋 Copied environment {env_name}@{original_config.version} to {new_name}@{new_version}"
            )
            return copied_config

        except Exception as e:
            logger.error(f"| ❌ Failed to copy environment: {e}")
            raise

    async def unregister(self, env_name: str) -> bool:
        """实现 `unregister` 的业务逻辑。"""
        if env_name not in self._environment_configs:
            logger.warning(f"| ⚠️ Environment {env_name} not found")
            return False

        env_config = self._environment_configs[env_name]

        # 配置相关参数。
        del self._environment_configs[env_name]

        # 注册相关组件。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract()

        logger.info(f"| 🗑️ Unregistered environment {env_name}@{env_config.version}")
        return True

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
                    "num_environments": len(self._environment_configs),
                    "num_versions": sum(
                        len(versions)
                        for versions in self._environment_history_versions.values()
                    ),
                },
                "environments": {},
            }

            for env_name, version_map in self._environment_history_versions.items():
                try:
                    versions_data: dict[str, dict[str, Any]] = {}
                    for env_config in version_map.values():
                        config_dict = env_config.model_dump()
                        versions_data[env_config.version] = config_dict

                    # 配置相关参数。
                    # 配置相关参数。
                    current_version = None
                    if env_name in self._environment_configs:
                        current_config = self._environment_configs[env_name]
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

                    save_data["environments"][env_name] = {
                        "versions": versions_data,
                        "current_version": current_version,
                    }
                except Exception as e:
                    logger.warning(
                        f"| ⚠️ Failed to serialize environment {env_name}: {e}"
                    )
                    continue

            # 持久化相关数据。
            await write_json_file(file_path, save_data)

            logger.info(
                f"| 💾 Saved {len(self._environment_configs)} environments with version history to {file_path}"
            )
            return str(file_path)

    async def load_from_json(
        self, file_path: str | None = None, auto_initialize: bool = True
    ) -> bool:
        """加载与 `load_from_json` 对应的数据或状态。"""

        file_path = file_path if file_path is not None else self.save_path

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Environment file not found: {file_path}")
                return False

            try:
                load_data = await read_json_file(file_path)

                environments_data = load_data.get("environments", {})
                loaded_count = 0

                for env_name, env_data in environments_data.items():
                    try:
                        # 配置相关参数。
                        versions_data = env_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(
                                f"| ⚠️ Environment {env_name} has invalid format for 'versions' (expected dict), skipping"
                            )
                            continue

                        current_version_str = env_data.get("current_version")

                        # 加载所需数据。
                        version_configs = []
                        latest_config = None
                        latest_version = None

                        for version_str, config_dict in versions_data.items():
                            # 处理版本与历史记录。
                            if "version" not in config_dict:
                                config_dict["version"] = version_str

                            try:
                                env_config = EnvironmentConfig.from_dict(config_dict)
                                version_configs.append(env_config)
                            except Exception as e:
                                logger.warning(
                                    f"| ⚠️ Failed to load environment config for {env_name}@{version_str}: {e}"
                                )
                                continue

                            # 处理版本与历史记录。
                            if (
                                latest_config is None
                                or (
                                    current_version_str
                                    and env_config.version == current_version_str
                                )
                                or (
                                    not current_version_str
                                    and (
                                        latest_version is None
                                        or version_manager.compare_versions(
                                            env_config.version, latest_version
                                        )
                                        > 0
                                    )
                                )
                            ):
                                latest_config = env_config
                                latest_version = env_config.version

                        # 处理版本与历史记录。
                        self._environment_history_versions[env_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }

                        # 更新相关状态。
                        if latest_config:
                            self._environment_configs[env_name] = latest_config

                            # 注册相关组件。
                            for env_config in version_configs:
                                await version_manager.register_version(
                                    "environment", env_name, env_config.version
                                )

                            # 持久化相关数据。
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)

                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load environment {env_name}: {e}")
                        continue

                logger.info(
                    f"| 📂 Loaded {loaded_count} environments with version history from {file_path}"
                )
                return True

            except Exception as e:
                logger.error(f"| ❌ Failed to load environments from {file_path}: {e}")
                return False

    async def restore(
        self, env_name: str, version: str, auto_initialize: bool = True
    ) -> EnvironmentConfig | None:
        """实现 `restore` 的业务逻辑。"""
        # 处理版本与历史记录。
        version_config = None
        if env_name in self._environment_history_versions:
            version_config = self._environment_history_versions[env_name].get(version)

        if version_config is None:
            logger.warning(
                f"| ⚠️ Version {version} not found for environment {env_name}"
            )
            return None

        # 创建所需对象。
        restored_config = EnvironmentConfig(**version_config.model_dump())

        # 配置相关参数。
        self._environment_configs[env_name] = restored_config

        # 更新相关状态。
        version_history = await version_manager.get_version_history(
            "environment", env_name
        )
        if version_history:
            # 注册相关组件。
            if version not in version_history.versions:
                await version_manager.register_version("environment", env_name, version)
            version_history.current_version = version
        else:
            # 注册相关组件。
            await version_manager.register_version("environment", env_name, version)

        # 初始化相关状态。
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)

        # 持久化相关数据。
        await self.save_to_json()

        logger.info(f"| 🔄 Restored environment {env_name} to version {version}")
        return restored_config

    async def save_contract(self, env_names: builtins.list[str] | None = None):
        """保存与 `save_contract` 对应的数据或状态。"""
        contract = []
        if env_names is not None:
            for index, env_name in enumerate(env_names):
                env_info = await self.get_info(env_name)
                if env_info is None:
                    continue
                text = env_info.rules
                contract.append(f"{index + 1:04d}\n{text}\n")
        else:
            for index, env_name in enumerate(self._environment_configs.keys()):
                env_info = await self.get_info(env_name)
                text = env_info.rules
                contract.append(f"{index + 1:04d}\n{text}\n")
        contract_text = "---\n".join(contract)
        await write_text_file(self.contract_path, contract_text)
        logger.info(
            f"| 📝 Saved {len(contract)} environments contract to {self.contract_path}"
        )

    async def load_contract(self) -> str:
        """加载与 `load_contract` 对应的数据或状态。"""
        return await read_text_file(self.contract_path)

    async def retrieve(self, query: str, k: int = 4) -> builtins.list[dict[str, Any]]:
        """实现 `retrieve` 的业务逻辑。"""
        if self._faiss_service is None:
            logger.warning(
                "| ⚠️ FAISS service not initialized, cannot retrieve environments"
            )
            return []

        try:
            from src.environment.faiss.types import FaissSearchRequest

            request = FaissSearchRequest(
                query=query,
                k=k,
                fetch_k=k * 5,  # 检索所需信息。
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
                    env_name = metadata.get("name", "")

                    # 配置相关参数。
                    env_config = None
                    if env_name and env_name in self._environment_configs:
                        env_config = self._environment_configs[env_name]

                    documents.append(
                        {
                            "name": env_name,
                            "description": metadata.get("description", ""),
                            "score": float(score),
                            "content": doc.get("page_content", "")
                            if isinstance(doc, dict)
                            else str(doc),
                            "config": env_config.model_dump() if env_config else None,
                        }
                    )

            return documents

        except Exception as e:
            logger.error(f"| ❌ Error retrieving environments: {e}")
            return []

    async def get_variables(self, env_name: str | None = None) -> dict[str, "Variable"]:
        """获取与 `get_variables` 对应的数据或状态。"""
        # 说明相关实现细节。
        from src.optimizer.types import Variable

        variables: dict[str, Variable] = {}

        if env_name is not None:
            # 说明相关实现细节。
            env_config = self._environment_configs.get(env_name)
            if env_config is None:
                logger.warning(f"| ⚠️ Environment {env_name} not found")
                return variables

            env_configs = {env_name: env_config}
        else:
            # 说明相关实现细节。
            env_configs = self._environment_configs

        for name, env_config in env_configs.items():
            # 说明相关实现细节。
            env_code = ""
            if env_config.cls is not None:
                env_code = dynamic_manager.get_full_module_source(env_config.cls) or ""
            elif env_config.code:
                env_code = env_config.code

            # 创建所需对象。
            variable = Variable(
                name=name,
                type="environment_code",
                description=env_config.description or f"Code for environment {name}",
                require_grad=env_config.require_grad,
                template=None,
                variables=env_code,  # 说明相关实现细节。
            )
            variables[name] = variable

        return variables

    async def get_trainable_variables(
        self, env_name: str | None = None
    ) -> dict[str, "Variable"]:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        async with self._variables_lock:
            all_variables = await self.get_variables(env_name=env_name)
            trainable_variables = {
                name: var for name, var in all_variables.items() if var.require_grad
            }
            return trainable_variables

    async def set_variables(
        self,
        env_name: str,
        variable_updates: dict[str, Any],
        new_version: str | None = None,
        description: str | None = None,
    ) -> EnvironmentConfig:
        """设置与 `set_variables` 对应的数据或状态。"""
        async with self._variables_lock:
            original_config = self._environment_configs.get(env_name)
            if original_config is None:
                raise ValueError(
                    f"Environment {env_name} not found. Use register() to register a new environment."
                )

            # 更新相关状态。
            # 说明相关实现细节。
            if "variables" not in variable_updates:
                raise ValueError(
                    f"variable_updates must contain 'variables' field with environment code, got: {list(variable_updates.keys())}"
                )

            new_code = variable_updates["variables"]
            if not isinstance(new_code, str):
                raise TypeError(
                    f"Environment code must be a string, got {type(new_code)}"
                )

            # 加载所需数据。
            class_name = dynamic_manager.extract_class_name_from_code(new_code)
            if not class_name:
                raise ValueError("Cannot extract class name from code")

            try:
                env_cls = dynamic_manager.load_class(
                    new_code,
                    class_name=class_name,
                    base_class=Environment,
                    context="environment",
                )
            except Exception as e:
                logger.error(f"| ❌ Failed to load environment class from code: {e}")
                raise ValueError(f"Failed to load environment class from code: {e}")

            # 持久化相关数据。
            # 创建所需对象。
            update_description = description or f"Updated code for {env_name}"
            return await self.update(
                env_cls=env_cls,
                env_config_dict=original_config.config,
                new_version=new_version,
                description=update_description,
                code=new_code,  # 创建所需对象。
            )

    async def cleanup(self):
        """释放组件占用的资源。"""
        try:
            # 清理并释放相关资源。
            for env_name, env_config in self._environment_configs.items():
                if env_config.instance and hasattr(env_config.instance, "cleanup"):
                    try:
                        await env_config.instance.cleanup()
                    except Exception as e:
                        logger.warning(
                            f"| ⚠️ Error cleaning up environment {env_name} instance: {e}"
                        )

            # 配置相关参数。
            self._environment_configs.clear()
            self._environment_history_versions.clear()

            # 执行异步任务。
            if self._faiss_service is not None:
                await self._faiss_service.cleanup()

            logger.info("| 🧹 Environment context manager cleaned up")

        except Exception as e:
            logger.error(f"| ❌ Error during environment context manager cleanup: {e}")

    async def __call__(
        self,
        name: str,
        action: str,
        input: dict[str, Any],
        ctx: SessionContext = None,
        **kwargs,
    ) -> Any:
        """执行组件调用并返回结果。"""
        if ctx is None:
            ctx = SessionContext()

        if name in self._environment_configs:
            env_config = self._environment_configs[name]

            version = env_config.version
            env_instance = env_config.instance
            logger.info(f"| ✅ Using environment {name}@{version}")

            action_config = env_config.actions.get(action)
            if action_config is None:
                raise ValueError(f"Action {action} not found in environment {name}")
            action_function = action_config.function

            # 说明相关实现细节。
            env_args = {
                "ctx": ctx,
            }

            # 加载所需数据。
            # 说明相关实现细节。
            if hasattr(action_function, "__self__"):
                # 说明相关实现细节。
                return await action_function(**input, **env_args)
            else:
                # 处理输入参数。
                return await action_function(env_instance, **input, **env_args)
        else:
            raise ValueError(f"Environment {name} not found")
