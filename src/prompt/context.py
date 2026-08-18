"""提供上下文管理相关实现。"""

import asyncio
import atexit
import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.optimizer.types import Variable

import builtins

from src.config import config
from src.dynamic import dynamic_manager
from src.logger import logger
from src.message.types import Message
from src.prompt.types import Prompt, PromptConfig
from src.registry import PROMPT
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


class PromptContextManager(BaseModel):
    """定义 `PromptContextManager`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(
        default=None, description="The base directory to use for the prompts"
    )
    save_path: str = Field(default=None, description="The path to save the prompts")
    contract_path: str = Field(
        default=None, description="The path to save the prompt contract"
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

        # 更新相关状态。
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(
                os.path.join(config.workdir, "prompt")
            )
        os.makedirs(self.base_dir, exist_ok=True)

        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "prompt.json")

        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")

        logger.info(
            f"| 📁 Prompt context manager base directory: {self.base_dir} and save path: {self.save_path}"
        )
        logger.info(f"| 📁 Prompt context manager contract path: {self.contract_path}")

        self._prompt_configs: dict[str, PromptConfig] = {}  # 配置相关参数。
        # 配置相关参数。
        self._prompt_history_versions: dict[str, dict[str, PromptConfig]] = {}

        self._cleanup_registered = False
        self._variables_lock = asyncio.Lock()  # 更新相关状态。

        # 清理并释放相关资源。
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True

    async def initialize(self, prompt_names: list[str] | None = None):
        """初始化组件及其依赖资源。"""
        # 加载所需数据。
        prompt_configs = {}
        registry_prompt_configs: dict[
            str, PromptConfig
        ] = await self._load_from_registry()
        prompt_configs.update(registry_prompt_configs)

        # 加载所需数据。
        code_prompt_configs: dict[str, PromptConfig] = await self._load_from_code()

        # 配置相关参数。
        for prompt_name, code_config in code_prompt_configs.items():
            if prompt_name in prompt_configs:
                registry_config = prompt_configs[prompt_name]
                # 处理版本与历史记录。
                if (
                    version_manager.compare_versions(
                        code_config.version, registry_config.version
                    )
                    > 0
                ):
                    logger.info(
                        f"| 🔄 Overriding prompt {prompt_name} from registry (v{registry_config.version}) with code version (v{code_config.version})"
                    )
                    prompt_configs[prompt_name] = code_config
                else:
                    logger.info(
                        f"| 📌 Keeping prompt {prompt_name} from registry (v{registry_config.version}), code version (v{code_config.version}) is not greater"
                    )
                    # 配置相关参数。
                    if (
                        version_manager.compare_versions(
                            code_config.version, registry_config.version
                        )
                        == 0
                        and prompt_name in self._prompt_history_versions
                    ):
                        self._prompt_history_versions[prompt_name][
                            registry_config.version
                        ] = registry_config
            else:
                # 说明相关实现细节。
                prompt_configs[prompt_name] = code_config

        # 说明相关实现细节。
        # 处理工具调用。
        if prompt_names is not None:
            filtered_configs = {}
            for base_name in prompt_names:
                # 说明相关实现细节。
                system_prompt_name = f"{base_name}_system_prompt"
                if system_prompt_name in prompt_configs:
                    filtered_configs[system_prompt_name] = prompt_configs[
                        system_prompt_name
                    ]
                # 说明相关实现细节。
                agent_message_prompt_name = f"{base_name}_agent_message_prompt"
                if agent_message_prompt_name in prompt_configs:
                    filtered_configs[agent_message_prompt_name] = prompt_configs[
                        agent_message_prompt_name
                    ]
            prompt_configs = filtered_configs

        # 说明相关实现细节。
        for prompt_name, prompt_config in prompt_configs.items():
            self._prompt_configs[prompt_name] = prompt_config

        # 创建所需对象。
        prompt_names = list(prompt_configs.keys())
        tasks = [self.build(prompt_configs[name]) for name in prompt_names]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )

        for prompt_name, result in zip(prompt_names, results):
            if isinstance(result, Exception):
                logger.error(
                    f"| ❌ Failed to initialize prompt {prompt_name}: {result}"
                )
                continue
            self._prompt_configs[prompt_name] = result
            logger.info(f"| 🔧 Prompt {prompt_name} initialized")

        # 配置相关参数。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract(prompt_names=prompt_names)

        logger.info("| ✅ Prompts initialization completed")

    async def _load_from_registry(self):
        """实现 `_load_from_registry` 的业务逻辑。"""
        prompt_configs: dict[str, PromptConfig] = {}

        async def register_prompt_class(prompt_cls: type[Prompt]):
            """注册与 `register_prompt_class` 对应的数据或状态。"""
            try:
                # 创建所需对象。
                prompt_instance = prompt_cls()

                # 初始化相关状态。
                if hasattr(prompt_instance, "initialize"):
                    await prompt_instance.initialize()

                # 配置相关参数。
                prompt_config_dict = prompt_instance.prompt_config
                if prompt_config_dict is None:
                    raise ValueError(
                        f"Prompt class {prompt_cls.__name__} must have 'prompt_config' field"
                    )

                # 配置相关参数。
                prompt_type = prompt_config_dict.get("type", prompt_instance.type)
                prompt_name = prompt_config_dict.get("name", prompt_instance.name)
                prompt_description = prompt_config_dict.get(
                    "description", prompt_instance.description
                )

                # 处理版本与历史记录。
                prompt_version = await version_manager.get_version(
                    "prompt", prompt_name
                )

                prompt_template = prompt_config_dict.get("template", "")
                prompt_variables = prompt_config_dict.get("variables", [])
                prompt_metadata = prompt_config_dict.get("metadata", {})

                # 说明相关实现细节。
                prompt_code = dynamic_manager.get_full_module_source(prompt_cls)

                # 配置相关参数。
                prompt_config = PromptConfig(
                    name=prompt_name,
                    type=prompt_type,
                    description=prompt_description,
                    version=prompt_version,
                    template=prompt_template,
                    variables=prompt_variables,
                    cls=prompt_cls,
                    instance=prompt_instance,  # 说明相关实现细节。
                    config={},
                    metadata=prompt_metadata,
                    code=prompt_code,
                )

                # 配置相关参数。
                prompt_configs[prompt_name] = prompt_config

                # 处理版本与历史记录。
                if prompt_name not in self._prompt_history_versions:
                    self._prompt_history_versions[prompt_name] = {}
                self._prompt_history_versions[prompt_name][prompt_version] = (
                    prompt_config
                )

                # 注册相关组件。
                await version_manager.register_version(
                    "prompt", prompt_name, prompt_version
                )

                logger.info(
                    f"| 📝 Registered prompt: {prompt_name} ({prompt_cls.__name__})"
                )

            except Exception as e:
                logger.error(
                    f"| ❌ Failed to register prompt class {prompt_cls.__name__}: {e}"
                )
                raise

        # 注册相关组件。
        prompt_classes = list(PROMPT._module_dict.values())

        logger.info(
            f"| 🔍 Discovering {len(prompt_classes)} prompts from PROMPT registry"
        )

        # 注册相关组件。
        tasks = [register_prompt_class(prompt_cls) for prompt_cls in prompt_classes]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )
        success_count = sum(1 for r in results if not isinstance(r, Exception))

        logger.info(
            f"| ✅ Discovered and registered {success_count}/{len(prompt_classes)} prompts from PROMPT registry"
        )

        return prompt_configs

    async def _load_from_code(self):
        """实现 `_load_from_code` 的业务逻辑。"""
        prompt_configs: dict[str, PromptConfig] = {}

        # 加载所需数据。
        if not os.path.exists(self.save_path):
            logger.info(
                f"| 📂 Prompt config file not found at {self.save_path}, skipping code-based loading"
            )
            return prompt_configs

        # 配置相关参数。
        try:
            load_data = await read_json_file(self.save_path)
        except json.JSONDecodeError as e:
            logger.warning(
                f"| ⚠️ Failed to parse prompt config JSON from {self.save_path}: {e}"
            )
            return prompt_configs

        prompts_data = load_data.get("prompts", {})

        async def register_prompt_class(
            prompt_name: str, prompt_data: dict[str, Any]
        ) -> tuple[str, dict[str, PromptConfig], PromptConfig | None] | None:
            """注册与 `register_prompt_class` 对应的数据或状态。"""
            try:
                current_version = prompt_data.get("current_version", "1.0.0")
                versions = prompt_data.get("versions", {})

                if not versions:
                    logger.warning(f"| ⚠️ Prompt {prompt_name} has no versions")
                    return None

                version_map: dict[str, PromptConfig] = {}
                current_config: PromptConfig | None = None  # 配置相关参数。

                for version_data in versions.values():
                    # 配置相关参数。
                    prompt_config = PromptConfig.from_dict(version_data)
                    version = prompt_config.version
                    version_map[version] = prompt_config

                    if version == current_version:
                        current_config = prompt_config

                return prompt_name, version_map, current_config
            except Exception as e:
                logger.error(
                    f"| ❌ Failed to load prompt {prompt_name} from code JSON: {e}"
                )
                return None

        # 加载所需数据。
        tasks = [
            register_prompt_class(prompt_name, prompt_data)
            for prompt_name, prompt_data in prompts_data.items()
        ]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            prompt_name, version_map, current_config = result
            if not version_map:
                continue

            # 处理版本与历史记录。
            self._prompt_history_versions[prompt_name] = version_map
            # 配置相关参数。
            if current_config is not None:
                prompt_configs[prompt_name] = current_config
            else:
                # 执行回退或重试逻辑。
                logger.warning(
                    f"| ⚠️ Prompt {prompt_name} current_version not found, using last available version"
                )
                prompt_configs[prompt_name] = list(version_map.values())[-1]

            # 注册相关组件。
            for prompt_config in version_map.values():
                await version_manager.register_version(
                    "prompt", prompt_name, prompt_config.version
                )

        logger.info(f"| 📂 Loaded {len(prompt_configs)} prompts from {self.save_path}")
        return prompt_configs

    async def register(
        self, prompt: Prompt | dict[str, Any], *, override: bool = False, **kwargs: Any
    ) -> PromptConfig:
        """实现 `register` 的业务逻辑。"""
        try:
            if isinstance(prompt, Prompt):
                prompt_name = prompt.name
                prompt_type = prompt.type
                prompt_description = prompt.description
                # 配置相关参数。
                prompt_config_dict = prompt.prompt_config
                if prompt_config_dict is None:
                    raise ValueError("Prompt instance must have 'prompt_config' field")
                prompt_template = prompt_config_dict.get("template", "")
                prompt_variables = prompt_config_dict.get("variables", [])
                prompt_cls = type(prompt)
            elif isinstance(prompt, dict):
                prompt_name = prompt.get("name")
                prompt_type = prompt.get("type", "prompt")
                prompt_description = prompt.get("description", "")
                prompt_template = prompt.get("template", "")
                prompt_variables = prompt.get("variables", [])
                prompt_cls = None
            else:
                raise TypeError(
                    f"Expected Prompt instance or dict, got {type(prompt)!r}"
                )

            if not prompt_name:
                raise ValueError("Prompt.name cannot be empty.")

            if prompt_name in self._prompt_configs and not override:
                raise ValueError(
                    f"Prompt '{prompt_name}' already registered. Use override=True to replace it."
                )

            # 处理版本与历史记录。
            version = await version_manager.get_version("prompt", prompt_name)

            # 说明相关实现细节。
            prompt_code = None
            if prompt_cls is not None:
                prompt_code = dynamic_manager.get_full_module_source(prompt_cls)

            # 配置相关参数。
            prompt_config = PromptConfig(
                name=prompt_name,
                type=prompt_type,
                description=prompt_description,
                version=version,
                template=prompt_template,
                variables=prompt_variables,
                cls=prompt_cls,
                instance=None,  # 说明相关实现细节。
                config=kwargs if kwargs else {},
                metadata=prompt.get("metadata", {}) if isinstance(prompt, dict) else {},
                code=prompt_code,
            )

            # 说明相关实现细节。
            self._prompt_configs[prompt_name] = prompt_config

            # 处理版本与历史记录。
            if prompt_name not in self._prompt_history_versions:
                self._prompt_history_versions[prompt_name] = {}
            self._prompt_history_versions[prompt_name][prompt_config.version] = (
                prompt_config
            )

            # 注册相关组件。
            await version_manager.register_version(
                "prompt", prompt_name, prompt_config.version
            )

            # 持久化相关数据。
            await self.save_to_json()
            # 持久化相关数据。
            await self.save_contract()

            logger.debug(
                f"| 📝 Registered prompt: {prompt_name} v{prompt_config.version}"
            )
            return prompt_config

        except Exception as e:
            logger.error(f"| ❌ Failed to register prompt: {e}")
            raise

    async def build(
        self, prompt_config: PromptConfig, force_rebuild: bool = False
    ) -> PromptConfig:
        """实现 `build` 的业务逻辑。"""
        if not force_rebuild and prompt_config.name in self._prompt_configs:
            existing_config = self._prompt_configs[prompt_config.name]
            if existing_config.instance is not None:
                return existing_config

        # 创建所需对象。
        try:
            # 加载所需数据。
            if prompt_config.cls is None:
                raise ValueError(
                    f"Cannot create prompt {prompt_config.name}: no class provided. Class should be loaded during initialization."
                )

            if prompt_config.instance is None or force_rebuild:
                # 配置相关参数。
                # 配置相关参数。
                instance_prompt_config = {
                    "name": prompt_config.name,
                    "type": prompt_config.type,
                    "description": prompt_config.description,
                    "template": prompt_config.template,
                    "variables": prompt_config.variables,
                    "metadata": prompt_config.metadata,
                }

                # 配置相关参数。
                init_kwargs = (
                    prompt_config.config.copy() if prompt_config.config else {}
                )
                init_kwargs["prompt_config"] = instance_prompt_config
                prompt_instance = prompt_config.cls(**init_kwargs)

                # 初始化相关状态。
                if hasattr(prompt_instance, "initialize"):
                    await prompt_instance.initialize()
            else:
                prompt_instance = prompt_config.instance

            prompt_config.instance = prompt_instance

            # 说明相关实现细节。
            self._prompt_configs[prompt_config.name] = prompt_config

            logger.info(f"| 🔧 Prompt {prompt_config.name} created and stored")

            return prompt_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create prompt {prompt_config.name}: {e}")
            raise

    async def update(
        self,
        prompt_name: str,
        prompt: Prompt | dict[str, Any],
        new_version: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> PromptConfig:
        """实现 `update` 的业务逻辑。"""
        original_config = self._prompt_configs.get(prompt_name)
        if original_config is None:
            raise ValueError(
                f"Prompt {prompt_name} not found. Use register() to register a new prompt."
            )

        # 说明相关实现细节。
        if isinstance(prompt, Prompt):
            new_description = prompt.description
            # 配置相关参数。
            prompt_config_dict = prompt.prompt_config
            if prompt_config_dict is None:
                raise ValueError("Prompt instance must have 'prompt_config' field")
            prompt_template = prompt_config_dict.get(
                "template", original_config.template
            )
            prompt_variables = prompt_config_dict.get(
                "variables", original_config.variables
            )
            prompt_cls = type(prompt)
            prompt_instance = prompt
        elif isinstance(prompt, dict):
            new_description = prompt.get("description", original_config.description)
            prompt_template = prompt.get("template", original_config.template)
            prompt_variables = prompt.get("variables", original_config.variables)
            # 说明相关实现细节。
            prompt_cls = original_config.cls
            prompt_instance = None
        else:
            raise TypeError(f"Expected Prompt instance or dict, got {type(prompt)!r}")

        # 处理版本与历史记录。
        if new_version is None:
            # 处理版本与历史记录。
            new_version = await version_manager.generate_next_version(
                "prompt", prompt_name, "patch"
            )

        # 说明相关实现细节。
        prompt_code = None
        if prompt_cls is not None:
            prompt_code = dynamic_manager.get_full_module_source(prompt_cls)
        elif original_config.code is not None:
            prompt_code = original_config.code

        # 配置相关参数。
        if prompt_instance is not None:
            updated_config = PromptConfig(
                name=prompt_name,
                type=original_config.type,
                description=description or new_description,
                version=new_version,
                template=prompt_template,
                variables=prompt_variables,
                cls=prompt_cls,
                config={},
                instance=None,  # 说明相关实现细节。
                metadata=prompt.get("metadata", {})
                if isinstance(prompt, dict)
                else original_config.metadata,
                code=prompt_code,
            )
        else:
            updated_config = PromptConfig(
                name=prompt_name,
                type=original_config.type,
                description=description or new_description,
                version=new_version,
                template=prompt_template,
                variables=prompt_variables,
                cls=prompt_cls,
                config=kwargs,
                instance=None,
                metadata=prompt.get("metadata", {})
                if isinstance(prompt, dict)
                else original_config.metadata,
                code=prompt_code,
            )

        # 配置相关参数。
        self._prompt_configs[prompt_name] = updated_config

        # 处理版本与历史记录。
        if prompt_name not in self._prompt_history_versions:
            self._prompt_history_versions[prompt_name] = {}
        self._prompt_history_versions[prompt_name][updated_config.version] = (
            updated_config
        )

        # 注册相关组件。
        await version_manager.register_version(
            "prompt",
            prompt_name,
            new_version,
            description=description or f"Updated from {original_config.version}",
        )

        # 创建所需对象。
        # 创建所需对象。
        try:
            if updated_config.cls is not None:
                updated_config = await self.build(updated_config, force_rebuild=True)
        except Exception as e:
            logger.warning(
                f"| ⚠️ Failed to build updated prompt instance for {prompt_name}: {e}"
            )

        # 持久化相关数据。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract()

        logger.info(f"| 📝 Updated prompt: {prompt_name} v{updated_config.version}")

        return updated_config

    async def copy(
        self,
        prompt_name: str,
        new_name: str | None = None,
        new_version: str | None = None,
        **override_config,
    ) -> PromptConfig:
        """实现 `copy` 的业务逻辑。"""
        original_config = self._prompt_configs.get(prompt_name)
        if original_config is None:
            raise ValueError(f"Prompt {prompt_name} not found")

        # 说明相关实现细节。
        if new_name is None:
            new_name = prompt_name

        # 处理版本与历史记录。
        if new_version is None:
            if new_name == prompt_name:
                # 处理版本与历史记录。
                new_version = await version_manager.generate_next_version(
                    "prompt", new_name, "patch"
                )
            else:
                # 处理版本与历史记录。
                new_version = await version_manager.get_or_generate_version(
                    "prompt", new_name
                )

        # 配置相关参数。
        new_config_dict = original_config.model_dump()
        new_config_dict["name"] = new_name
        new_config_dict["version"] = new_version

        # 说明相关实现细节。
        if override_config:
            if "description" in override_config:
                new_config_dict["description"] = override_config.pop("description")
            if "metadata" in override_config:
                new_config_dict["metadata"].update(override_config.pop("metadata"))
            # 配置相关参数。
            new_config_dict["config"].update(override_config)

        # 创建所需对象。
        new_config_dict["instance"] = None

        # 加载所需数据。
        new_config = PromptConfig.from_dict(new_config_dict)

        # 注册相关组件。
        self._prompt_configs[new_name] = new_config

        # 处理版本与历史记录。
        if new_name not in self._prompt_history_versions:
            self._prompt_history_versions[new_name] = {}
        self._prompt_history_versions[new_name][new_version] = new_config

        # 注册相关组件。
        await version_manager.register_version(
            "prompt",
            new_name,
            new_version,
            description=f"Copied from {prompt_name}@{original_config.version}",
        )

        # 持久化相关数据。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract()

        logger.info(
            f"| 📋 Copied prompt {prompt_name}@{original_config.version} to {new_name}@{new_version}"
        )
        return new_config

    async def unregister(self, prompt_name: str) -> bool:
        """实现 `unregister` 的业务逻辑。"""
        if prompt_name not in self._prompt_configs:
            logger.warning(f"| ⚠️ Prompt {prompt_name} not found")
            return False

        prompt_config = self._prompt_configs[prompt_name]

        # 配置相关参数。
        del self._prompt_configs[prompt_name]

        # 注册相关组件。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract()

        logger.info(f"| 🗑️ Unregistered prompt {prompt_name}@{prompt_config.version}")
        return True

    async def restore(
        self, prompt_name: str, version: str, auto_initialize: bool = True
    ) -> PromptConfig | None:
        """实现 `restore` 的业务逻辑。"""
        # 处理版本与历史记录。
        version_config = None
        if prompt_name in self._prompt_history_versions:
            version_config = self._prompt_history_versions[prompt_name].get(version)

        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for prompt {prompt_name}")
            return None

        # 创建所需对象。
        # 加载所需数据。
        restored_config = PromptConfig.from_dict(version_config.model_dump())

        # 配置相关参数。
        self._prompt_configs[prompt_name] = restored_config

        # 更新相关状态。
        version_history = await version_manager.get_version_history(
            "prompt", prompt_name
        )
        if version_history:
            # 注册相关组件。
            if version not in version_history.versions:
                await version_manager.register_version("prompt", prompt_name, version)
            version_history.current_version = version
        else:
            # 注册相关组件。
            await version_manager.register_version("prompt", prompt_name, version)

        # 初始化相关状态。
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)

        # 持久化相关数据。
        await self.save_to_json()
        # 持久化相关数据。
        await self.save_contract()

        logger.info(f"| 🔄 Restored prompt {prompt_name} to version {version}")
        return restored_config

    async def get(self, name: str) -> Prompt | None:
        """实现 `get` 的业务逻辑。"""
        prompt_config = self._prompt_configs.get(name)
        if prompt_config is None:
            return None
        return prompt_config.instance if prompt_config.instance is not None else None

    async def get_info(self, name: str) -> PromptConfig | None:
        """获取与 `get_info` 对应的数据或状态。"""
        return self._prompt_configs.get(name)

    async def list(self) -> list[str]:
        """实现 `list` 的业务逻辑。"""
        return [name for name in self._prompt_configs]

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
                    "num_prompts": len(self._prompt_configs),
                    "num_versions": sum(
                        len(versions)
                        for versions in self._prompt_history_versions.values()
                    ),
                },
                "prompts": {},
            }

            for prompt_name, version_map in self._prompt_history_versions.items():
                try:
                    # 配置相关参数。
                    versions_data: dict[str, dict[str, Any]] = {}
                    for version_str, prompt_config in version_map.items():
                        # 配置相关参数。
                        config_dict = prompt_config.model_dump()

                        # 处理版本与历史记录。
                        versions_data[prompt_config.version] = config_dict

                    # 配置相关参数。
                    current_version = None
                    if prompt_name in self._prompt_configs:
                        current_config = self._prompt_configs[prompt_name]
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

                    save_data["prompts"][prompt_name] = {
                        "versions": versions_data,
                        "current_version": current_version,
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize prompt {prompt_name}: {e}")
                    continue

            # 持久化相关数据。
            await write_json_file(file_path, save_data)

            logger.info(
                f"| 💾 Saved {len(self._prompt_configs)} prompts with version history to {file_path}"
            )
            return str(file_path)

    async def load_from_json(
        self, file_path: str | None = None, auto_initialize: bool = True
    ) -> bool:
        """加载与 `load_from_json` 对应的数据或状态。"""
        file_path = file_path if file_path is not None else self.save_path

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Prompt file not found: {file_path}")
                return False

            try:
                load_data = await read_json_file(file_path)

                prompts_data = load_data.get("prompts", {})
                loaded_count = 0

                for prompt_name, prompt_data in prompts_data.items():
                    try:
                        # 配置相关参数。
                        versions_data = prompt_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(
                                f"| ⚠️ Prompt {prompt_name} has invalid format for 'versions' (expected dict), skipping"
                            )
                            continue

                        current_version_str = prompt_data.get("current_version")

                        # 加载所需数据。
                        version_configs = []
                        latest_config = None
                        latest_version = None

                        for version_str, config_dict in versions_data.items():
                            # 处理版本与历史记录。
                            if "version" not in config_dict:
                                config_dict["version"] = version_str

                            # 配置相关参数。
                            prompt_config = PromptConfig.from_dict(config_dict)

                            version_configs.append(prompt_config)

                            # 处理版本与历史记录。
                            if (
                                latest_config is None
                                or (
                                    current_version_str
                                    and prompt_config.version == current_version_str
                                )
                                or (
                                    not current_version_str
                                    and (
                                        latest_version is None
                                        or version_manager.compare_versions(
                                            prompt_config.version, latest_version
                                        )
                                        > 0
                                    )
                                )
                            ):
                                latest_config = prompt_config
                                latest_version = prompt_config.version

                        # 处理版本与历史记录。
                        self._prompt_history_versions[prompt_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }

                        # 更新相关状态。
                        if latest_config:
                            self._prompt_configs[prompt_name] = latest_config

                            # 注册相关组件。
                            for prompt_config in version_configs:
                                await version_manager.register_version(
                                    "prompt", prompt_name, prompt_config.version
                                )

                            # 持久化相关数据。
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)

                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load prompt {prompt_name}: {e}")
                        continue

                logger.info(
                    f"| 📂 Loaded {loaded_count} prompts with version history from {file_path}"
                )
                return True

            except Exception as e:
                logger.error(f"| ❌ Failed to load prompts from {file_path}: {e}")
                return False

    async def get_system_message(
        self,
        prompt_name: str | None = None,
        modules: dict[str, Any] | None = None,
        reload: bool = False,
        **kwargs,
    ):
        """获取与 `get_system_message` 对应的数据或状态。"""
        prompt_name = f"{prompt_name}_system_prompt"
        prompt_info = await self.get_info(prompt_name)

        version = prompt_info.version
        prompt_instance = prompt_info.instance
        logger.info(f"| ✅ Using prompt {prompt_name}@{version}")

        return await prompt_instance.get_message(modules, reload, **kwargs)

    async def get_agent_message(
        self,
        prompt_name: str | None = None,
        modules: dict[str, Any] | None = None,
        reload: bool = True,
        **kwargs,
    ):
        """获取与 `get_agent_message` 对应的数据或状态。"""
        prompt_name = f"{prompt_name}_agent_message_prompt"

        prompt_info = await self.get_info(prompt_name)
        version = prompt_info.version

        prompt_instance = prompt_info.instance
        logger.info(f"| ✅ Using prompt {prompt_name}@{version}")

        return await prompt_instance.get_message(modules, reload, **kwargs)

    async def get_messages(
        self,
        prompt_name: str | None = None,
        system_modules: dict[str, Any] | None = None,
        agent_modules: dict[str, Any] | None = None,
        **kwargs,
    ) -> builtins.list[Message]:
        """获取与 `get_messages` 对应的数据或状态。"""
        system_message = await self.get_system_message(
            prompt_name, system_modules, reload=False, **kwargs
        )
        agent_message = await self.get_agent_message(
            prompt_name, agent_modules, reload=True, **kwargs
        )
        return [system_message, agent_message]

    async def get_variables(
        self, prompt_name: str | None = None
    ) -> dict[str, "Variable"]:
        """获取与 `get_variables` 对应的数据或状态。"""
        system_prompt_name = f"{prompt_name}_system_prompt"
        agent_prompt_name = f"{prompt_name}_agent_message_prompt"
        system_prompt_instance = await self.get(system_prompt_name)
        agent_prompt_instance = await self.get(agent_prompt_name)
        variables = {}

        if system_prompt_instance is not None:
            system_var = await system_prompt_instance.get_variable()
            variables[system_prompt_name] = system_var

        if agent_prompt_instance is not None:
            agent_var = await agent_prompt_instance.get_variable()
            variables[agent_prompt_name] = agent_var

        return variables

    async def get_trainable_variables(
        self, prompt_name: str | None = None
    ) -> dict[str, "Variable"]:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        async with self._variables_lock:
            system_prompt_name = f"{prompt_name}_system_prompt"
            agent_prompt_name = f"{prompt_name}_agent_message_prompt"
            system_prompt_instance = await self.get(system_prompt_name)
            agent_prompt_instance = await self.get(agent_prompt_name)
            variables: dict[str, Variable] = {}

            # 说明相关实现细节。
            if system_prompt_instance is not None:
                system_var = await system_prompt_instance.get_variable()
                trainable_sub_vars = system_var.get_trainable_variables()
                for var_name, var in trainable_sub_vars.items():
                    variables[var_name] = var
                    logger.debug(
                        f"| ✅ Extracted trainable variable '{var_name}' from {system_prompt_name}"
                    )
            else:
                logger.debug(
                    f"| ⚠️ System prompt instance {system_prompt_name} not found"
                )

            # 说明相关实现细节。
            if agent_prompt_instance is not None:
                agent_var = await agent_prompt_instance.get_variable()
                trainable_sub_vars = agent_var.get_trainable_variables()
                for var_name, var in trainable_sub_vars.items():
                    variables[var_name] = var
                    logger.debug(
                        f"| ✅ Extracted trainable variable '{var_name}' from {agent_prompt_name}"
                    )
            else:
                logger.debug(f"| ⚠️ Agent prompt instance {agent_prompt_name} not found")

            return variables

    async def set_variables(
        self,
        prompt_name: str,
        variable_updates: dict[str, Any],
        new_version: str | None = None,
        description: str | None = None,
    ) -> dict[str, PromptConfig]:
        """设置与 `set_variables` 对应的数据或状态。"""
        async with self._variables_lock:
            import copy

            system_prompt_name = f"{prompt_name}_system_prompt"
            agent_prompt_name = f"{prompt_name}_agent_message_prompt"

            # 说明相关实现细节。
            system_prompt_instance = await self.get(system_prompt_name)
            agent_prompt_instance = await self.get(agent_prompt_name)

            # 更新相关状态。
            system_updates = {}
            agent_updates = {}

            # 更新相关状态。
            for var_name, new_value in variable_updates.items():
                found = False

                # 校验输入与当前状态。
                if system_prompt_instance is not None:
                    system_config = self._prompt_configs.get(system_prompt_name)
                    if (
                        system_config
                        and isinstance(system_config.variables, dict)
                        and var_name in system_config.variables
                    ):
                        system_updates[var_name] = new_value
                        found = True
                        logger.debug(
                            f"| 📍 Variable '{var_name}' belongs to {system_prompt_name}"
                        )

                # 校验输入与当前状态。
                if not found and agent_prompt_instance is not None:
                    agent_config = self._prompt_configs.get(agent_prompt_name)
                    if (
                        agent_config
                        and isinstance(agent_config.variables, dict)
                        and var_name in agent_config.variables
                    ):
                        agent_updates[var_name] = new_value
                        found = True
                        logger.debug(
                            f"| 📍 Variable '{var_name}' belongs to {agent_prompt_name}"
                        )

                if not found:
                    logger.warning(f"| ⚠️ Variable '{var_name}' not found in any prompt")

            # 更新相关状态。
            updated_configs = {}

            # 更新相关状态。
            if system_updates:
                system_config = self._prompt_configs.get(system_prompt_name)
                updated_variables = copy.deepcopy(system_config.variables)

                for var_name, new_value in system_updates.items():
                    if isinstance(updated_variables[var_name], dict):
                        updated_variables[var_name]["variables"] = new_value
                        logger.info(
                            f"| ✅ Updated variable '{var_name}' in {system_prompt_name}"
                        )
                    elif hasattr(updated_variables[var_name], "variables"):
                        updated_variables[var_name].variables = new_value
                        logger.info(
                            f"| ✅ Updated variable '{var_name}' in {system_prompt_name}"
                        )

                prompt_dict = system_config.model_dump()
                prompt_dict["variables"] = updated_variables

                updated_config = await self.update(
                    prompt_name=system_prompt_name,
                    prompt=prompt_dict,
                    new_version=new_version,
                    description=description,
                )
                # 加载所需数据。
                try:
                    if (
                        updated_config
                        and getattr(updated_config, "cls", None) is not None
                        and updated_config.instance is None
                    ):
                        await self.build(updated_config, force_rebuild=True)
                except Exception as e:
                    logger.warning(
                        f"| ⚠️ Failed to build updated prompt instance for {system_prompt_name}: {e}"
                    )
                updated_configs[system_prompt_name] = updated_config

            # 更新相关状态。
            if agent_updates:
                agent_config = self._prompt_configs.get(agent_prompt_name)
                updated_variables = copy.deepcopy(agent_config.variables)

                for var_name, new_value in agent_updates.items():
                    if isinstance(updated_variables[var_name], dict):
                        updated_variables[var_name]["variables"] = new_value
                        logger.info(
                            f"| ✅ Updated variable '{var_name}' in {agent_prompt_name}"
                        )
                    elif hasattr(updated_variables[var_name], "variables"):
                        updated_variables[var_name].variables = new_value
                        logger.info(
                            f"| ✅ Updated variable '{var_name}' in {agent_prompt_name}"
                        )

                prompt_dict = agent_config.model_dump()
                prompt_dict["variables"] = updated_variables

                updated_config = await self.update(
                    prompt_name=agent_prompt_name,
                    prompt=prompt_dict,
                    new_version=new_version,
                    description=description,
                )
                # 加载所需数据。
                try:
                    if (
                        updated_config
                        and getattr(updated_config, "cls", None) is not None
                        and updated_config.instance is None
                    ):
                        await self.build(updated_config, force_rebuild=True)
                except Exception as e:
                    logger.warning(
                        f"| ⚠️ Failed to build updated prompt instance for {agent_prompt_name}: {e}"
                    )
                updated_configs[agent_prompt_name] = updated_config

            if not updated_configs:
                raise ValueError(
                    f"No variables were updated. Check variable names: {list(variable_updates.keys())}"
                )

            logger.info(
                f"| ✅ Updated {len(system_updates) + len(agent_updates)} variables across {len(updated_configs)} prompts"
            )
            return updated_configs

    async def save_contract(self, prompt_names: builtins.list[str] | None = None):
        """保存与 `save_contract` 对应的数据或状态。"""
        contract = []
        if prompt_names is not None:
            # 说明相关实现细节。
            filtered_names = []
            for base_name in prompt_names:
                system_prompt_name = f"{base_name}_system_prompt"
                agent_prompt_name = f"{base_name}_agent_message_prompt"
                if system_prompt_name in self._prompt_configs:
                    filtered_names.append(system_prompt_name)
                if agent_prompt_name in self._prompt_configs:
                    filtered_names.append(agent_prompt_name)

            for index, prompt_name in enumerate(filtered_names):
                prompt_info = await self.get_info(prompt_name)
                if prompt_info:
                    # 转换并规范化数据。
                    contract_text = f"Prompt: {prompt_info.name}\nType: {prompt_info.type}\nDescription: {prompt_info.description}\nTemplate:\n{prompt_info.template}\n"
                    contract.append(f"{index + 1:04d}\n{contract_text}\n")
        else:
            for index, prompt_name in enumerate(self._prompt_configs.keys()):
                prompt_info = await self.get_info(prompt_name)
                if prompt_info:
                    # 转换并规范化数据。
                    contract_text = f"Prompt: {prompt_info.name}\nType: {prompt_info.type}\nDescription: {prompt_info.description}\nTemplate:\n{prompt_info.template}\n"
                    contract.append(f"{index + 1:04d}\n{contract_text}\n")

        contract_text = "---\n".join(contract)
        await write_text_file(self.contract_path, contract_text)
        logger.info(
            f"| 📝 Saved {len(contract)} prompts contract to {self.contract_path}"
        )

    async def load_contract(self) -> str:
        """加载与 `load_contract` 对应的数据或状态。"""
        return await read_text_file(self.contract_path)

    async def cleanup(self):
        """释放组件占用的资源。"""
        try:
            # 配置相关参数。
            self._prompt_configs.clear()
            self._prompt_history_versions.clear()
            logger.info("| 🧹 Prompt context manager cleaned up")

        except Exception as e:
            logger.error(f"| ❌ Error during prompt context manager cleanup: {e}")
