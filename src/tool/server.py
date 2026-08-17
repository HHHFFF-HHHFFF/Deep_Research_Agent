"""提供服务入口相关实现。"""
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
import asyncio
import os
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from src.logger import logger
from src.config import config
from src.tool.context import ToolContextManager
from src.tool.types import Tool, ToolConfig, ToolResponse
from src.session import SessionContext
from src.utils import assemble_project_path

class TCPServer(BaseModel):
    """定义 `TCPServer`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the tools")
    save_path: str = Field(default=None, description="The path to save the tools")
    contract_path: str = Field(default=None, description="The path to save the tool contract")

    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """初始化实例。"""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, ToolConfig] = {}  # 配置相关参数。


    async def initialize(self, tool_names: Optional[List[str]] = None):
        """初始化组件及其依赖资源。"""

        self.base_dir = assemble_project_path(os.path.join(config.workdir, "tool"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "tool.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 TCP Server base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")

        # 初始化相关状态。
        self.tool_context_manager = ToolContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
            model_name="openrouter/gemini-3-flash-preview",
            embedding_model_name="openrouter/text-embedding-3-large",
        )
        await self.tool_context_manager.initialize(tool_names=tool_names)

        logger.info("| ✅ Tools initialization completed")

    async def get_contract(self) -> str:
        """获取与 `get_contract` 对应的数据或状态。"""
        return await self.tool_context_manager.load_contract()

    async def register(self,
                       tool: Union[Tool, Type[Tool]],
                       config: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None,
                       code: Optional[str] = None) -> ToolConfig:
        """实现 `register` 的业务逻辑。"""
        tool_config = await self.tool_context_manager.register(
            tool,
            tool_config_dict=config,
            override=override,
            version=version,
            code=code
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config

    async def list(self) -> List[str]:
        """实现 `list` 的业务逻辑。"""
        return await self.tool_context_manager.list()


    async def get(self, tool_name: str) -> Tool:
        """实现 `get` 的业务逻辑。"""
        tool = await self.tool_context_manager.get(tool_name)
        return tool

    async def get_info(self, tool_name: str) -> Optional[ToolConfig]:
        """获取与 `get_info` 对应的数据或状态。"""
        return await self.tool_context_manager.get_info(tool_name)

    async def cleanup(self):
        """释放组件占用的资源。"""
        await self.tool_context_manager.cleanup()
        self._registered_configs.clear()

    async def update(self,
                     tool_name: str, tool: Union[Tool, Type[Tool]],
                     config: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None,
                     description: Optional[str] = None) -> ToolConfig:
        """实现 `update` 的业务逻辑。"""
        tool_config = await self.tool_context_manager.update(
            tool_name, tool, tool_config_dict=config, new_version=new_version, description=description
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config

    async def copy(self, tool_name: str, new_name: Optional[str] = None,
                  new_version: Optional[str] = None, **override_config) -> ToolConfig:
        """实现 `copy` 的业务逻辑。"""
        tool_config = await self.tool_context_manager.copy(
            tool_name, new_name, new_version, **override_config
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config

    async def unregister(self, tool_name: str) -> bool:
        """实现 `unregister` 的业务逻辑。"""
        success = await self.tool_context_manager.unregister(tool_name)
        if success and tool_name in self._registered_configs:
            del self._registered_configs[tool_name]
        return success

    async def restore(self, tool_name: str, version: str, auto_initialize: bool = True) -> Optional[ToolConfig]:
        """实现 `restore` 的业务逻辑。"""
        tool_config = await self.tool_context_manager.restore(tool_name, version, auto_initialize)
        if tool_config:
            self._registered_configs[tool_config.name] = tool_config
        return tool_config

    async def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """实现 `retrieve` 的业务逻辑。"""
        return await self.tool_context_manager.retrieve(query=query, k=k)

    async def get_variables(self, tool_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_variables` 对应的数据或状态。"""
        return await self.tool_context_manager.get_variables(tool_name=tool_name)

    async def get_trainable_variables(self, tool_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        return await self.tool_context_manager.get_trainable_variables(tool_name=tool_name)

    async def set_variables(self, tool_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> ToolConfig:
        """设置与 `set_variables` 对应的数据或状态。"""
        updated_config = await self.tool_context_manager.set_variables(
            tool_name=tool_name,
            variable_updates=variable_updates,
            new_version=new_version,
            description=description
        )
        self._registered_configs[updated_config.name] = updated_config
        return updated_config

    async def __call__(self,
                       name: str,
                       input: Dict[str, Any],
                       timeout: Optional[float] = None,
                       ctx: SessionContext = None,
                       **kwargs
                       ) -> ToolResponse:
        """执行组件调用并返回结果。"""
        return await self.tool_context_manager(name,
                                               input,
                                               timeout=timeout,
                                               ctx=ctx,
                                               **kwargs)


# 说明相关实现细节。
tcp = TCPServer()
