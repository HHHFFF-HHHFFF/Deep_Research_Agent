"""提供服务入口相关实现。"""

import os
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.logger import logger
from src.agent.types import AgentConfig, Agent
from src.agent.context import AgentContextManager
from src.utils import assemble_project_path

class ACPServer(BaseModel):
    """定义 `ACPServer`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the agents")
    save_path: str = Field(default=None, description="The path to save the agents")
    contract_path: str = Field(default=None, description="The path to save the agent contract")

    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """初始化实例。"""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, AgentConfig] = {}  # 配置相关参数。


    async def initialize(self, agent_names: Optional[List[str]] = None):
        """初始化组件及其依赖资源。"""

        self.base_dir = assemble_project_path(os.path.join(config.workdir, "agent"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "agent.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 ACP Server base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")

        # 初始化相关状态。
        self.agent_context_manager = AgentContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
            model_name="openrouter/gemini-3-flash-preview",
            embedding_model_name="openrouter/text-embedding-3-large",
        )
        await self.agent_context_manager.initialize(agent_names=agent_names)

        # 配置相关参数。
        agent_list = await self.agent_context_manager.list()
        for agent_name in agent_list:
            agent_config = await self.agent_context_manager.get_info(agent_name)
            if agent_config and agent_name not in self._registered_configs:
                self._registered_configs[agent_name] = agent_config

        logger.info("| ✅ Agents initialization completed")

    async def get_contract(self) -> str:
        """获取与 `get_contract` 对应的数据或状态。"""
        return await self.agent_context_manager.load_contract()

    async def register(self,
                       agent_cls: Type[Agent],
                       agent_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> AgentConfig:
        """实现 `register` 的业务逻辑。"""
        agent_config = await self.agent_context_manager.register(
            agent_cls,
            agent_config_dict=agent_config_dict,
            override=override,
            version=version
        )
        self._registered_configs[agent_config.name] = agent_config
        return agent_config

    async def get_info(self, agent_name: str) -> Optional[AgentConfig]:
        """获取与 `get_info` 对应的数据或状态。"""
        return await self.agent_context_manager.get_info(agent_name)

    async def list(self) -> List[str]:
        """实现 `list` 的业务逻辑。"""
        return await self.agent_context_manager.list()


    async def get(self, agent_name: str) -> Optional[Agent]:
        """实现 `get` 的业务逻辑。"""
        agent = await self.agent_context_manager.get(agent_name)
        return agent

    async def cleanup(self):
        """释放组件占用的资源。"""
        await self.agent_context_manager.cleanup()
        self._registered_configs.clear()

    async def update(self,
                     agent_cls: Type[Agent],
                     agent_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None,
                     description: Optional[str] = None) -> AgentConfig:
        """实现 `update` 的业务逻辑。"""
        agent_config = await self.agent_context_manager.update(
            agent_cls, agent_config_dict=agent_config_dict, new_version=new_version, description=description
        )
        self._registered_configs[agent_config.name] = agent_config
        return agent_config

    async def copy(self,
                  agent_name: str,
                  new_name: Optional[str] = None,
                  new_version: Optional[str] = None,
                  new_config: Optional[Dict[str, Any]] = None) -> AgentConfig:
        """实现 `copy` 的业务逻辑。"""
        agent_config = await self.agent_context_manager.copy(
            agent_name, new_name, new_version, new_config
        )
        self._registered_configs[agent_config.name] = agent_config
        return agent_config

    async def unregister(self, agent_name: str) -> bool:
        """实现 `unregister` 的业务逻辑。"""
        success = await self.agent_context_manager.unregister(agent_name)
        if success and agent_name in self._registered_configs:
            del self._registered_configs[agent_name]
        return success

    async def restore(self, agent_name: str, version: str, auto_initialize: bool = True) -> Optional[AgentConfig]:
        """实现 `restore` 的业务逻辑。"""
        agent_config = await self.agent_context_manager.restore(agent_name, version, auto_initialize)
        if agent_config:
            self._registered_configs[agent_config.name] = agent_config
        return agent_config

    async def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """实现 `retrieve` 的业务逻辑。"""
        return await self.agent_context_manager.retrieve(query=query, k=k)

    async def get_variables(self, agent_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_variables` 对应的数据或状态。"""
        return await self.agent_context_manager.get_variables(agent_name=agent_name)

    async def get_trainable_variables(self, agent_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        return await self.agent_context_manager.get_trainable_variables(agent_name=agent_name)

    async def set_variables(self, agent_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> AgentConfig:
        """设置与 `set_variables` 对应的数据或状态。"""
        updated_config = await self.agent_context_manager.set_variables(
            agent_name=agent_name,
            variable_updates=variable_updates,
            new_version=new_version,
            description=description
        )
        self._registered_configs[updated_config.name] = updated_config
        return updated_config

    async def __call__(self, name: str, input: Dict[str, Any], **kwargs) -> Any:
        """执行组件调用并返回结果。"""
        return await self.agent_context_manager(name, input, **kwargs)


# 说明相关实现细节。
acp = ACPServer()
