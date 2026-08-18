"""提供数据类型相关实现。"""

import builtins
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field

from src.dynamic import dynamic_manager
from src.logger import logger
from src.message import HumanMessage, Message, SystemMessage
from src.optimizer.types import Variable


class Prompt(BaseModel):
    """定义 `Prompt`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(
        description="The type of the prompt, e.g. 'system_prompt' or 'agent_message_prompt'"
    )
    name: str = Field(description="The name of the prompt")
    description: str = Field(description="The description of the prompt")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="The metadata of the prompt"
    )
    prompt_config: dict[str, Any] | None = Field(
        default=None, description="The prompt information"
    )

    prompt_variable: Variable | None = Field(
        default=None, description="The prompt variable"
    )
    message: Message | None = Field(default=None, description="The message")

    def __init__(self, prompt_config: dict[str, Any] | None = None, **kwargs):
        """初始化实例。"""
        super().__init__(**kwargs)
        self.prompt_config = (
            prompt_config if prompt_config is not None else self.prompt_config
        )

    async def initialize(self) -> None:
        """初始化组件及其依赖资源。"""

    async def _load_prompt_variable(self) -> None:
        """实现 `_load_prompt_variable` 的业务逻辑。"""
        if self.prompt_variable is not None:
            return

        if self.prompt_config is None:
            raise ValueError("Cannot load prompt: prompt_config is None")

        try:
            self.prompt_variable = Variable.from_dict(self.prompt_config)
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt: {e}")

    async def get_variable(self, reload: bool = False) -> Variable:
        """获取与 `get_variable` 对应的数据或状态。"""
        if self.prompt_variable is None or reload:
            await self._load_prompt_variable()
        if self.prompt_variable is None:
            raise RuntimeError("提示词变量加载完成后仍为空")
        return self.prompt_variable

    async def get_trainable_variable(self) -> dict[str, Variable]:
        """获取与 `get_trainable_variable` 对应的数据或状态。"""
        prompt_variable = await self.get_variable()
        return prompt_variable.get_trainable_variables()

    async def get_message(
        self, modules: dict[str, Any] | None = None, reload: bool = False, **kwargs
    ):
        """获取与 `get_message` 对应的数据或状态。"""
        # 加载所需数据。
        prompt_variable = await self.get_variable(reload=reload)

        is_system_prompt = self.type == "system_prompt"

        # 校验输入与当前状态。
        if is_system_prompt and not reload and self.message is not None:
            return self.message

        try:
            # 创建所需对象。
            if modules is None or len(modules) == 0:
                modules = prompt_variable.get_modules()
            else:
                # 说明相关实现细节。
                variable_modules = prompt_variable.get_modules()
                modules = {**variable_modules, **modules}

            prompt_str = prompt_variable.render(modules)

            # 组装并返回结果。
            if is_system_prompt:
                self.message = SystemMessage(content=prompt_str)
            else:
                # 说明相关实现细节。
                self.message = HumanMessage(content=prompt_str)

            return self.message

        except Exception as e:
            logger.warning(f"Failed to render prompt: {e}")
            raise RuntimeError(f"Failed to render prompt: {e}")

    def __str__(self):
        return f"Prompt(name={self.name}, description={self.description})"

    def __repr__(self):
        return self.__str__()


class PromptConfig(BaseModel):
    """定义 `PromptConfig`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the prompt")
    type: str = Field(description="The type of the prompt")
    description: str = Field(description="The description of the prompt")
    version: str = Field(default="1.0.0", description="Version of the prompt")
    template: str = Field(description="The template string for the prompt")
    variables: Union[dict[str, "Variable"], "Variable"] | None = Field(
        default=None,
        description="The variables used in the template. Can be Dict[str, Variable] or single Variable",
    )
    cls: builtins.type[Prompt] | None = Field(
        default=None, description="The class of the prompt"
    )
    instance: Any | None = Field(default=None, description="The instance of the prompt")
    config: dict[str, Any] | None = Field(
        default_factory=dict,
        description="The initialization configuration of the prompt",
    )
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="The metadata of the prompt"
    )
    code: str | None = Field(
        default=None,
        description="Source code for dynamically generated prompt classes (used when cls cannot be imported from a module)",
    )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """实现 `model_dump` 的业务逻辑。"""

        def serialize_variables(vars_data: Any) -> Any:
            """序列化与 `serialize_variables` 对应的数据或状态。"""
            if vars_data is None:
                return None
            elif isinstance(vars_data, Variable):
                # 转换并规范化数据。
                return {
                    "name": vars_data.name,
                    "type": vars_data.type,
                    "description": vars_data.description,
                    "require_grad": vars_data.require_grad,
                    "template": vars_data.template,
                    "variables": serialize_variables(vars_data.variables),
                }
            elif isinstance(vars_data, dict):
                # 说明相关实现细节。
                return {k: serialize_variables(v) for k, v in vars_data.items()}
            elif isinstance(vars_data, (list, tuple)):
                # 说明相关实现细节。
                return [serialize_variables(item) for item in vars_data]
            else:
                # 组装并返回结果。
                return vars_data

        result = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "version": self.version,
            "template": self.template,
            "variables": serialize_variables(self.variables),
            "metadata": self.metadata,
            "config": self.config,
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "instance": None,
            "code": self.code,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptConfig":
        """从持久化字典恢复提示词配置，不执行其中携带的代码。"""
        payload = dict(data)
        payload["cls"] = None
        payload["instance"] = None
        return cls.model_validate(payload)

    def __str__(self):
        return f"PromptConfig(name={self.name}, type={self.type}, description={self.description}, version={self.version})"

    def __repr__(self):
        return self.__str__()
