"""提供服务入口相关实现。"""
import os
from typing import Any, Dict, List, Optional, Union, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.utils import assemble_project_path
from src.logger import logger
from src.memory.types import MemoryConfig, Memory
from src.session import SessionContext
from src.memory.context import MemoryContextManager

class MemoryManager(BaseModel):
    """定义 `MemoryManager`，封装相关数据与行为。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="The base directory to use for the memory systems")
    save_path: str = Field(default=None, description="The path to save the memory systems")
    contract_path: str = Field(default=None, description="The path to save the memory contract")

    def __init__(self, **kwargs):
        """初始化实例。"""
        super().__init__(**kwargs)
        self._registered_memories: Dict[str, MemoryConfig] = {}  # 配置相关参数。

    async def initialize(self, memory_names: Optional[List[str]] = None):
        """初始化组件及其依赖资源。"""
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "memory"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "memory.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Memory Manager base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")

        # 初始化相关状态。
        self.memory_context_manager = MemoryContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path
        )
        await self.memory_context_manager.initialize(memory_names=memory_names)

        logger.info("| ✅ Memory systems initialization completed")

    async def register(self, memory: Union[Memory, Type[Memory]], *, override: bool = False, **kwargs: Any) -> MemoryConfig:
        """实现 `register` 的业务逻辑。"""
        memory_config = await self.memory_context_manager.register(memory, override=override, **kwargs)
        self._registered_memories[memory_config.name] = memory_config
        return memory_config

    async def update(self, memory_name: str, memory: Union[Memory, Type[Memory]],
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> MemoryConfig:
        """实现 `update` 的业务逻辑。"""
        memory_config = await self.memory_context_manager.update(memory_name, memory, new_version, description, **kwargs)
        self._registered_memories[memory_config.name] = memory_config
        return memory_config

    async def get_info(self, memory_name: str) -> Optional[MemoryConfig]:
        """获取与 `get_info` 对应的数据或状态。"""
        return await self.memory_context_manager.get_info(memory_name)

    async def list(self) -> List[str]:
        """实现 `list` 的业务逻辑。"""
        return await self.memory_context_manager.list()

    async def get_contract(self) -> str:
        """获取与 `get_contract` 对应的数据或状态。"""
        return await self.memory_context_manager.load_contract()

    async def get(self, memory_name: str) -> Memory:
        """实现 `get` 的业务逻辑。"""
        return await self.memory_context_manager.get(memory_name)

    async def cleanup(self):
        """释放组件占用的资源。"""
        if hasattr(self, 'memory_context_manager'):
            await self.memory_context_manager.cleanup()

    async def start_session(self,
                            memory_name: str,
                            agent_name: Optional[str] = None,
                            task_id: Optional[str] = None,
                            description: Optional[str] = None,
                            ctx: SessionContext = None,
                            **kwargs) -> str:
        """实现 `start_session` 的业务逻辑。"""
        return await self.memory_context_manager.start_session(
            memory_name,
            agent_name,
            task_id,
            description,
            ctx=ctx,
            **kwargs
        )

    async def add_event(self,
                        memory_name: str,
                        step_number: int,
                        event_type: Any, data: Any,
                        agent_name: str,
                        task_id: Optional[str] = None,
                        ctx: SessionContext = None,
                        **kwargs):
        """添加与 `add_event` 对应的数据或状态。"""
        return await self.memory_context_manager.add_event(
            memory_name,
            step_number,
            event_type,
            data,
            agent_name,
            task_id,
            ctx=ctx,
        **kwargs)

    async def end_session(self,
                          memory_name: str,
                          ctx: SessionContext = None,
                          **kwargs):
        """实现 `end_session` 的业务逻辑。"""
        return await self.memory_context_manager.end_session(
            memory_name,
            ctx=ctx,
        **kwargs)

    async def get_session_info(self,
                               memory_name: str,
                               ctx: SessionContext = None,
                               **kwargs):
        """获取与 `get_session_info` 对应的数据或状态。"""
        return await self.memory_context_manager.get_session_info(memory_name, ctx=ctx, **kwargs)

    async def clear_session(self,
                            memory_name: str,
                            ctx: SessionContext = None,
                            **kwargs):
        """实现 `clear_session` 的业务逻辑。"""
        return await self.memory_context_manager.clear_session(memory_name, ctx=ctx, **kwargs)

    async def get_state(self,
                        name: str,
                        n: Optional[int] = None,
                        ctx: SessionContext = None,
                        **kwargs
                        ) -> Dict[str, Any]:
        """获取与 `get_state` 对应的数据或状态。"""
        return await self.memory_context_manager.get_state(name, n, ctx, **kwargs)

    async def get_variables(self, memory_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_variables` 对应的数据或状态。"""
        return await self.memory_context_manager.get_variables(memory_name=memory_name)

    async def get_trainable_variables(self, memory_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        return await self.memory_context_manager.get_trainable_variables(memory_name=memory_name)

    async def set_variables(self, memory_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> MemoryConfig:
        """设置与 `set_variables` 对应的数据或状态。"""
        updated_config = await self.memory_context_manager.set_variables(
            memory_name=memory_name,
            variable_updates=variable_updates,
            new_version=new_version,
            description=description
        )
        self._registered_memories[updated_config.name] = updated_config
        return updated_config


# 处理记忆或缓存状态。
memory_manager = MemoryManager()
