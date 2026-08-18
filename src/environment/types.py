"""提供数据类型相关实现。"""

import json
import uuid
from collections.abc import Callable
from enum import Enum
from typing import Any

import inflection
from pydantic import BaseModel, ConfigDict, Field

from src.dynamic import dynamic_manager


class Environment(BaseModel):
    """定义 `Environment`，封装相关数据与行为。"""

    name: str = Field(description="The name of the environment.")
    description: str = Field(description="The description of the environment.")
    metadata: dict[str, Any] = Field(description="The metadata of the environment.")
    require_grad: bool = Field(
        default=False, description="Whether the environment requires gradients"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init_subclass__(cls, **kwargs):
        """实现 `__init_subclass__` 的业务逻辑。"""
        super().__init_subclass__(**kwargs)
        # 初始化相关状态。

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化相关状态。
        self.actions: dict[str, ActionConfig] = {}

        # 注册相关组件。
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_action_name"):
                action_name = attr._action_name
                if action_name not in self.actions:
                    action_config = ActionConfig(
                        env_name=self.name,
                        name=action_name,
                        description=getattr(attr, "_action_description", ""),
                        function=attr,
                        metadata=getattr(attr, "_metadata", {}),
                    )
                    # 说明相关实现细节。
                    self.actions[action_name] = action_config

    async def get_state(self) -> dict[str, Any]:
        """获取与 `get_state` 对应的数据或状态。"""
        raise NotImplementedError("Get state method not implemented")

    def get_rules(self) -> str:
        """获取与 `get_rules` 对应的数据或状态。"""
        metadata = self.metadata if self.metadata else {}
        has_vision = metadata.get("has_vision", False)
        additional_rules = metadata.get("additional_rules", None)
        env_name = self.name
        actions = self.actions

        # 创建所需对象。
        rules_parts = [f"<environment_{inflection.underscore(env_name)}>"]

        # 说明相关实现细节。
        rules_parts.append("<state>")
        if additional_rules and "state" in additional_rules:
            rules_parts.append(additional_rules["state"])
        else:
            rules_parts.append(f"The environment state about {env_name}.")
        rules_parts.append("</state>")

        # 说明相关实现细节。
        rules_parts.append("<vision>")
        if additional_rules and "vision" in additional_rules:
            rules_parts.append(additional_rules["vision"])
        else:
            if has_vision:
                rules_parts.append("The environment vision information.")
            else:
                rules_parts.append("No vision available.")
        rules_parts.append("</vision>")

        # 说明相关实现细节。
        if additional_rules and "additional_rules" in additional_rules:
            rules_parts.append("<additional_rules>")
            rules_parts.append(additional_rules["additional_rules"])
            rules_parts.append("</additional_rules>")

        # 处理工具调用。
        rules_parts.append("<interaction>")

        if additional_rules and "interaction" in additional_rules:
            # 处理工具调用。
            rules_parts.append(additional_rules["interaction"])
        else:
            # 处理工具调用。
            rules_parts.append("Available actions:")

            # 组装并返回结果。
            sorted_actions = sorted(actions.items(), key=lambda x: x[0])

            for i, (action_name, action_config) in enumerate(sorted_actions, 1):
                rules_parts.append(f"{i}. {action_name}: {action_config.description}")

            rules_parts.append(
                "Input format: JSON string with action-specific parameters."
            )
            rules_parts.append(
                'Example: {"name": "action_name", "args": {"action-specific parameters"}}'
            )

        rules_parts.append("</interaction>")

        # 清理并释放相关资源。
        rules_parts.append(f"</environment_{inflection.underscore(env_name)}>")

        return "\n".join(rules_parts)


class ECPErrorCode(Enum):
    """定义 `ECPErrorCode`，封装相关数据与行为。"""

    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    ENVIRONMENT_NOT_FOUND = -32001
    ACTION_NOT_FOUND = -32002
    ACTION_EXECUTION_ERROR = -32003


class ECPError(BaseModel):
    """定义 `ECPError`，封装相关数据与行为。"""

    code: ECPErrorCode
    message: str
    data: dict[str, Any] | None = None


class ECPRequest(BaseModel):
    """定义 `ECPRequest`，封装相关数据与行为。"""

    id: str | int = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: dict[str, Any] | None = None


class ECPResponse(BaseModel):
    """定义 `ECPResponse`，封装相关数据与行为。"""

    id: str | int
    result: dict[str, Any] | None = None
    error: ECPError | None = None


class ECPNotification(BaseModel):
    """定义 `ECPNotification`，封装相关数据与行为。"""

    method: str
    params: dict[str, Any] | None = None


class ActionConfig(BaseModel):
    """定义 `ActionConfig`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    env_name: str = Field(
        description="The name of the environment this action belongs to"
    )
    name: str = Field(description="The name of the action")
    description: str = Field(description="The description of the action")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="The metadata of the action"
    )
    version: str = Field(default="1.0.0", description="Version of the action")

    function: Callable | None = Field(
        default=None, description="The function implementing the action"
    )
    code: str | None = Field(default=None, description="The source code of the action")

    # 说明相关实现细节。
    args_schema: type[BaseModel] | None = Field(
        default=None, description="Default args schema (BaseModel type)"
    )
    function_calling: dict[str, Any] | None = Field(
        default=None, description="Default function calling representation"
    )
    text: str | None = Field(default=None, description="Default text representation")

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """实现 `model_dump` 的业务逻辑。"""

        result = {
            "env_name": self.env_name,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "version": self.version,
            "function": f"<{self.function.__name__}>",
            "code": self.code,
            "args_schema": dynamic_manager.serialize_args_schema(self.args_schema)
            if self.args_schema
            else None,
            "function_calling": self.function_calling,
            "text": self.text,
        }

        return result

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "ActionConfig":
        """实现 `model_validate` 的业务逻辑。"""
        env_name = data.get("env_name")
        name = data.get("name")
        description = data.get("description")
        metadata = data.get("metadata")
        version = data.get("version")

        code = data.get("code")
        function = None

        args_schema = dynamic_manager.deserialize_args_schema(data.get("args_schema"))
        function_calling = data.get("function_calling")
        text = data.get("text")

        return cls(
            env_name=env_name,
            name=name,
            description=description,
            metadata=metadata,
            version=version,
            function=function,
            code=code,
            args_schema=args_schema,
            function_calling=function_calling,
            text=text,
        )


class EnvironmentConfig(BaseModel):
    """定义 `EnvironmentConfig`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the environment")
    description: str = Field(description="The description of the environment")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="The metadata of the environment"
    )
    rules: str = Field(description="The rules of the environment")
    version: str = Field(default="1.0.0", description="Version of the environment")
    require_grad: bool = Field(
        default=False, description="Whether the environment requires gradients"
    )

    cls: type[Environment] | None = Field(
        default=None, description="The class of the environment"
    )
    config: dict[str, Any] | None = Field(
        default={}, description="The initialization configuration of the environment"
    )
    instance: Any | None = Field(
        default=None, description="The instance of the environment"
    )
    code: str | None = Field(
        default=None,
        description="Source code for dynamically generated environment classes (used when cls cannot be imported from a module)",
    )

    actions: dict[str, ActionConfig] = Field(
        default_factory=dict,
        description="Dictionary of actions available in this environment",
    )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """实现 `model_dump` 的业务逻辑。"""
        result = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "rules": self.rules,
            "version": self.version,
            "require_grad": self.require_grad,
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "config": self.config,
            "instance": None,
            "code": self.code,
            "actions": {
                name: action_config.model_dump()
                for name, action_config in self.actions.items()
            },
        }

        return result

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "EnvironmentConfig":
        """实现 `model_validate` 的业务逻辑。"""

        name = data.get("name")
        description = data.get("description")
        metadata = data.get("metadata")
        rules = data.get("rules")
        version = data.get("version")
        require_grad = data.get("require_grad", False)

        cls_ = None
        code = data.get("code")
        if code:
            class_name = dynamic_manager.extract_class_name_from_code(code)
            if class_name:
                try:
                    cls_ = dynamic_manager.load_class(
                        code,
                        class_name=class_name,
                        base_class=Environment,
                        context="environment",
                    )
                except Exception:
                    cls_ = None
            else:
                cls_ = None
        else:
            cls_ = None

        config = data.get("config")
        instance = data.get("instance", None)

        actions = {
            name: ActionConfig.model_validate(action_config)
            for name, action_config in data.get("actions", {}).items()
        }

        # 加载所需数据。
        if cls_ is not None:
            for action_name, action_config in actions.items():
                # 处理工具调用。
                if hasattr(cls_, action_name):
                    attr = getattr(cls_, action_name)
                    if (
                        hasattr(attr, "_action_name")
                        and attr._action_name == action_name
                    ):
                        action_config.function = attr
                        continue

        return cls(
            name=name,
            description=description,
            metadata=metadata,
            rules=rules,
            version=version,
            require_grad=require_grad,
            cls=cls_,
            config=config,
            instance=instance,
            code=code,
            actions=actions,
        )


class ScreenshotInfo(BaseModel):
    """定义 `ScreenshotInfo`，封装相关数据与行为。"""

    transformed: bool = Field(
        default=False, description="Whether the screenshot has been transformed"
    )
    screenshot: str = Field(default="Screenshot base64")
    screenshot_path: str = Field(default="Screenshot path")
    screenshot_description: str = Field(default="Screenshot description")
    transform_info: dict[str, Any] | None = Field(
        default=None, description="Transform information"
    )


class ActionResult(BaseModel):
    """定义 `ActionResult`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    success: bool = Field(description="Whether the action was successful")
    message: str = Field(description="The message of the action result")
    extra: dict[str, Any] | None = Field(
        default=None, description="The extra information of the action result"
    )

    def __str__(self) -> str:
        return f"ActionResult(success={self.success}, message={self.message}, extra={self.extra})"

    def __repr__(self) -> str:
        return self.__str__()

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """实现 `model_dump` 的业务逻辑。"""
        from pydantic import BaseModel

        def serialize_value(value: Any) -> Any:
            if isinstance(value, BaseModel):
                return value.model_dump(**kwargs)
            if isinstance(value, list):
                return [serialize_value(item) for item in value]
            if isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            return value

        return {
            "success": self.success,
            "message": self.message,
            "extra": serialize_value(self.extra) if self.extra is not None else None,
        }

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump())


class EnvironmentState(BaseModel):
    """定义 `EnvironmentState`，封装相关数据与行为。"""

    state: str = Field(default="State", description="The state of the environment")
    extra: dict[str, Any] | None = Field(
        default=None, description="The extra information of the state"
    )
