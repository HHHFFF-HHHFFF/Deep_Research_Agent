"""提供服务入口相关实现。"""

import builtins
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.logger import logger
from src.session import SessionContext
from src.skill.context import SkillContextManager
from src.skill.types import SkillConfig, SkillResponse
from src.utils import assemble_project_path


class SCPServer(BaseModel):
    """定义 `SCPServer`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="Base directory for skill data")
    save_path: str = Field(default=None, description="Path to persist skill configs")
    contract_path: str = Field(default=None, description="Path to save skill contract")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.skill_context_manager: SkillContextManager | None = None

    # ------------------------------------------------------------------
    # 说明相关实现细节。
    # ------------------------------------------------------------------

    async def initialize(self, skill_names: list[str] | None = None):
        """初始化组件及其依赖资源。"""
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "skill"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "skill.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(
            f"| 📁 SCP Server base directory: {self.base_dir} "
            f"with save path: {self.save_path} and contract path: {self.contract_path}"
        )

        self.skill_context_manager = SkillContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
        )
        await self.skill_context_manager.initialize(skill_names=skill_names)

        logger.info("| ✅ Skills initialization completed")

    async def cleanup(self):
        """释放组件占用的资源。"""
        if self.skill_context_manager is not None:
            await self.skill_context_manager.cleanup()

    # ------------------------------------------------------------------
    # 注册相关组件。
    # ------------------------------------------------------------------

    async def register(
        self,
        skill_dir: str,
        override: bool = False,
        version: str | None = None,
    ) -> SkillConfig:
        """实现 `register` 的业务逻辑。"""
        return await self.skill_context_manager.register(
            skill_dir=skill_dir,
            override=override,
            version=version,
        )

    async def update(
        self,
        name: str,
        skill_dir: str | None = None,
        new_version: str | None = None,
        description: str | None = None,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SkillConfig:
        """实现 `update` 的业务逻辑。"""
        return await self.skill_context_manager.update(
            name=name,
            skill_dir=skill_dir,
            new_version=new_version,
            description=description,
            content=content,
            metadata=metadata,
        )

    async def unregister(self, name: str) -> bool:
        """实现 `unregister` 的业务逻辑。"""
        return await self.skill_context_manager.unregister(name)

    async def copy(
        self,
        name: str,
        new_name: str | None = None,
        new_version: str | None = None,
        new_skill_dir: str | None = None,
    ) -> SkillConfig:
        """实现 `copy` 的业务逻辑。"""
        return await self.skill_context_manager.copy(
            name=name,
            new_name=new_name,
            new_version=new_version,
            new_skill_dir=new_skill_dir,
        )

    async def restore(self, name: str, version: str) -> SkillConfig | None:
        """实现 `restore` 的业务逻辑。"""
        return await self.skill_context_manager.restore(name, version)

    # ------------------------------------------------------------------
    # 检索所需信息。
    # ------------------------------------------------------------------

    async def get(self, skill_name: str) -> SkillConfig | None:
        """实现 `get` 的业务逻辑。"""
        return await self.skill_context_manager.get(skill_name)

    async def get_info(self, skill_name: str) -> SkillConfig | None:
        """获取与 `get_info` 对应的数据或状态。"""
        return await self.skill_context_manager.get_info(skill_name)

    async def list(self) -> list[str]:
        """实现 `list` 的业务逻辑。"""
        return await self.skill_context_manager.list()

    # ------------------------------------------------------------------
    # 说明相关实现细节。
    # ------------------------------------------------------------------

    async def get_context(self, skill_names: builtins.list[str] | None = None) -> str:
        """获取与 `get_context` 对应的数据或状态。"""
        return await self.skill_context_manager.get_context(skill_names=skill_names)

    async def get_contract(self) -> str:
        """获取与 `get_contract` 对应的数据或状态。"""
        return await self.skill_context_manager.load_contract()

    # ------------------------------------------------------------------
    # 说明相关实现细节。
    # ------------------------------------------------------------------

    async def __call__(
        self,
        name: str,
        input: dict[str, Any],
        model_name: str | None = None,
        ctx: SessionContext = None,
        **kwargs,
    ) -> SkillResponse:
        """执行组件调用并返回结果。"""
        return await self.skill_context_manager(
            name=name,
            input=input,
            model_name=model_name,
            ctx=ctx,
            **kwargs,
        )


# 说明相关实现细节。
scp = SCPServer()
