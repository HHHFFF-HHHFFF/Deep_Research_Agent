"""提供服务入口相关实现。"""
from typing import Any, Dict, List, Optional, Type, Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.optimizer.types import Variable

import os
from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.config import config
from src.environment.context import EnvironmentContextManager
from src.environment.types import Environment, EnvironmentConfig
from src.session import SessionContext
from src.utils import assemble_project_path

class ECPServer(BaseModel):
    """定义 `ECPServer`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the environments")
    save_path: str = Field(default=None, description="The path to save the environments")
    contract_path: str = Field(default=None, description="The path to save the environment contract")

    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """初始化实例。"""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, EnvironmentConfig] = {}  # 配置相关参数。


    async def initialize(self, env_names: Optional[List[str]] = None):
        """初始化组件及其依赖资源。"""

        self.base_dir = assemble_project_path(os.path.join(config.workdir, "environment"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "environment.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 ECP Server base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")

        # 初始化相关状态。
        self.environment_context_manager = EnvironmentContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
            model_name="openrouter/gemini-3-flash-preview",
            embedding_model_name="openrouter/text-embedding-3-large"
        )
        await self.environment_context_manager.initialize(env_names=env_names)

        logger.info("| ✅ Environments initialization completed")

    async def get_contract(self) -> str:
        """获取与 `get_contract` 对应的数据或状态。"""
        return await self.environment_context_manager.load_contract()

    def action(self,
               name: str = None,
               description: str = "",
               metadata: Optional[Dict[str, Any]] = None):
        """实现 `action` 的业务逻辑。"""
        def decorator(func: Callable):
            action_name = name or func.__name__

            func._action_name = action_name
            func._action_description = description
            func._action_function = func
            func._action_metadata = metadata if metadata is not None else {}

            return func
        return decorator

    async def register(self,
                       env_cls: Type[Environment],
                       env_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> EnvironmentConfig:
        """实现 `register` 的业务逻辑。"""
        env_config = await self.environment_context_manager.register(
            env_cls,
            env_config_dict=env_config_dict,
            override=override,
            version=version
        )
        self._registered_configs[env_config.name] = env_config
        return env_config

    async def list(self) -> List[str]:
        """实现 `list` 的业务逻辑。"""
        return await self.environment_context_manager.list()


    async def get(self, env_name: str) -> Optional[Environment]:
        """实现 `get` 的业务逻辑。"""
        return await self.environment_context_manager.get(env_name)

    async def get_info(self, env_name: str) -> Optional[EnvironmentConfig]:
        """获取与 `get_info` 对应的数据或状态。"""
        return await self.environment_context_manager.get_info(env_name)

    async def get_state(self, env_name: str, ctx: SessionContext = None, **kwargs) -> Optional[Dict[str, Any]]:
        """获取与 `get_state` 对应的数据或状态。"""
        return await self.environment_context_manager.get_state(env_name, ctx, **kwargs)

    async def cleanup(self):
        """释放组件占用的资源。"""
        await self.environment_context_manager.cleanup()
        self._registered_configs.clear()

    async def update(self,
                     env_cls: Type[Environment],
                     env_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None,
                     description: Optional[str] = None) -> EnvironmentConfig:
        """实现 `update` 的业务逻辑。"""
        env_config = await self.environment_context_manager.update(
            env_cls, env_config_dict=env_config_dict, new_version=new_version, description=description
        )
        self._registered_configs[env_config.name] = env_config
        return env_config

    async def copy(self,
                  env_name: str,
                  new_name: Optional[str] = None,
                  new_version: Optional[str] = None,
                  new_config: Optional[Dict[str, Any]] = None) -> EnvironmentConfig:
        """实现 `copy` 的业务逻辑。"""
        env_config = await self.environment_context_manager.copy(
            env_name, new_name, new_version, new_config
        )
        self._registered_configs[env_config.name] = env_config
        return env_config

    async def unregister(self, env_name: str) -> bool:
        """实现 `unregister` 的业务逻辑。"""
        success = await self.environment_context_manager.unregister(env_name)
        if success and env_name in self._registered_configs:
            del self._registered_configs[env_name]
        return success

    async def restore(self, env_name: str, version: str, auto_initialize: bool = True) -> Optional[EnvironmentConfig]:
        """实现 `restore` 的业务逻辑。"""
        env_config = await self.environment_context_manager.restore(env_name, version, auto_initialize)
        if env_config:
            self._registered_configs[env_config.name] = env_config
        return env_config

    async def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """实现 `retrieve` 的业务逻辑。"""
        return await self.environment_context_manager.retrieve(query=query, k=k)

    async def get_variables(self, env_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_variables` 对应的数据或状态。"""
        return await self.environment_context_manager.get_variables(env_name=env_name)

    async def get_trainable_variables(self, env_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        return await self.environment_context_manager.get_trainable_variables(env_name=env_name)

    async def set_variables(self, env_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> EnvironmentConfig:
        """设置与 `set_variables` 对应的数据或状态。"""
        updated_config = await self.environment_context_manager.set_variables(
            env_name=env_name,
            variable_updates=variable_updates,
            new_version=new_version,
            description=description
        )
        self._registered_configs[updated_config.name] = updated_config
        return updated_config

    async def __call__(self,
                       name: str,
                       action: str,
                       input: Dict[str, Any],
                       ctx: SessionContext = None,
                       **kwargs) -> Any:
        """执行组件调用并返回结果。"""
        return await self.environment_context_manager(name, action, input, ctx, **kwargs)


# 说明相关实现细节。
ecp = ECPServer()
