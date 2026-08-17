"""提供服务入口相关实现。"""

import os
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.utils import assemble_project_path
from src.logger import logger
from src.prompt.types import PromptConfig, Prompt
from src.prompt.context import PromptContextManager
from src.message.types import Message
from src.optimizer.types import Variable

class PromptManager(BaseModel):
    """定义 `PromptManager`，封装相关数据与行为。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="The base directory to use for the prompts")
    save_path: str = Field(default=None, description="The path to save the prompts")
    contract_path: str = Field(default=None, description="The path to save the prompt contract")

    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """初始化实例。"""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, PromptConfig] = {}  # 配置相关参数。

    async def initialize(self, prompt_names: Optional[List[str]] = None):
        """初始化组件及其依赖资源。"""
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "prompt"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "prompt.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Prompt Manager base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")

        # 初始化相关状态。
        self.prompt_context_manager = PromptContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
        )
        await self.prompt_context_manager.initialize(prompt_names=prompt_names)

        logger.info("| ✅ Prompts initialization completed")

    async def register(self, prompt: Union[Prompt, Dict[str, Any]], *, override: bool = False, **kwargs: Any) -> PromptConfig:
        """实现 `register` 的业务逻辑。"""
        prompt_config = await self.prompt_context_manager.register(prompt, override=override, **kwargs)
        self._registered_configs[prompt_config.name] = prompt_config
        return prompt_config

    async def list(self) -> List[str]:
        """实现 `list` 的业务逻辑。"""
        return await self.prompt_context_manager.list()

    async def get(self, prompt_name: str) -> Optional[Prompt]:
        """实现 `get` 的业务逻辑。"""
        return await self.prompt_context_manager.get(prompt_name)

    async def get_info(self, prompt_name: str) -> Optional[PromptConfig]:
        """获取与 `get_info` 对应的数据或状态。"""
        return await self.prompt_context_manager.get_info(prompt_name)

    async def cleanup(self):
        """释放组件占用的资源。"""
        if hasattr(self, 'prompt_context_manager'):
            await self.prompt_context_manager.cleanup()
        self._registered_configs.clear()

    async def update(self, prompt_name: str, prompt: Union[Prompt, Dict[str, Any]],
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> PromptConfig:
        """实现 `update` 的业务逻辑。"""
        prompt_config = await self.prompt_context_manager.update(
            prompt_name, prompt, new_version=new_version, description=description, **kwargs
        )
        self._registered_configs[prompt_config.name] = prompt_config
        return prompt_config

    async def copy(self, prompt_name: str, new_name: Optional[str] = None,
                  new_version: Optional[str] = None, **override_config) -> PromptConfig:
        """实现 `copy` 的业务逻辑。"""
        prompt_config = await self.prompt_context_manager.copy(
            prompt_name, new_name, new_version, **override_config
        )
        self._registered_configs[prompt_config.name] = prompt_config
        return prompt_config

    async def unregister(self, prompt_name: str) -> bool:
        """实现 `unregister` 的业务逻辑。"""
        success = await self.prompt_context_manager.unregister(prompt_name)
        if success and prompt_name in self._registered_configs:
            del self._registered_configs[prompt_name]
        return success

    async def restore(self, prompt_name: str, version: str, auto_initialize: bool = True) -> Optional[PromptConfig]:
        """实现 `restore` 的业务逻辑。"""
        prompt_config = await self.prompt_context_manager.restore(prompt_name, version, auto_initialize)
        if prompt_config:
            self._registered_configs[prompt_config.name] = prompt_config
        return prompt_config

    async def get_system_message(self,
                          prompt_name: Optional[str] = None,
                          modules: Dict[str, Any] = None,
                          reload: bool = False,
                          **kwargs):
        """获取与 `get_system_message` 对应的数据或状态。"""
        # 初始化相关状态。
        if not hasattr(self, 'prompt_context_manager'):
            await self.initialize()

        return await self.prompt_context_manager.get_system_message(
            prompt_name=prompt_name,
            modules=modules,
            reload=reload,
            **kwargs
        )

    async def get_agent_message(self,
                         prompt_name: Optional[str] = None,
                         modules: Dict[str, Any] = None,
                         reload: bool = True,
                         **kwargs):
        """获取与 `get_agent_message` 对应的数据或状态。"""
        # 初始化相关状态。
        if not hasattr(self, 'prompt_context_manager'):
            await self.initialize()

        return await self.prompt_context_manager.get_agent_message(
            prompt_name=prompt_name,
            modules=modules,
            reload=reload,
            **kwargs
        )

    async def get_messages(self,
        prompt_name: Optional[str] = None,
        system_modules: Dict[str, Any] = None,
        agent_modules: Dict[str, Any] = None,
        **kwargs
    ) -> List[Message]:
        """获取与 `get_messages` 对应的数据或状态。"""
        return await self.prompt_context_manager.get_messages(
            prompt_name=prompt_name,
            system_modules=system_modules,
            agent_modules=agent_modules,
            **kwargs
        )

    async def get_variables(self, prompt_name: Optional[str] = None) -> Dict[str, Variable]:
        """获取与 `get_variables` 对应的数据或状态。"""
        return await self.prompt_context_manager.get_variables(prompt_name=prompt_name)

    async def set_variables(self, prompt_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> Dict[str, PromptConfig]:
        """设置与 `set_variables` 对应的数据或状态。"""
        # 配置相关参数。
        updated_configs = await self.prompt_context_manager.set_variables(
            prompt_name=prompt_name,
            variable_updates=variable_updates,
            new_version=new_version,
            description=description,
        )
        # 配置相关参数。
        for _, updated_config in updated_configs.items():
            self._registered_configs[updated_config.name] = updated_config
        return updated_configs

    async def get_trainable_variables(self, prompt_name: Optional[str] = None) -> Dict[str, Variable]:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        return await self.prompt_context_manager.get_trainable_variables(prompt_name=prompt_name)

    async def get_contract(self) -> str:
        """获取与 `get_contract` 对应的数据或状态。"""
        return await self.prompt_context_manager.load_contract()


# 说明相关实现细节。
prompt_manager = PromptManager()
