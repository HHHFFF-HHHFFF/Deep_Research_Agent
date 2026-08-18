import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.dynamic import dynamic_manager
from src.utils import dedent


class EventType(Enum):
    # 说明相关实现细节。
    TASK_START = "task_start"
    TOOL_STEP = "tool_step"
    TASK_END = "task_end"

    # 说明相关实现细节。
    OPTIMIZATION_STEP = "optimization_step"


class ChatEvent(BaseModel):
    id: str = Field(..., description="The unique identifier for the event.")
    step_number: int = Field(..., description="The step number of the event.")
    event_type: EventType = Field(..., description="The type of the event.")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="The timestamp of the event."
    )
    data: dict[str, Any] = Field(
        default_factory=dict, description="The data of the event."
    )
    agent_name: str | None = Field(
        None, description="The name of the agent that generated the event."
    )
    session_id: str | None = Field(None, description="The session ID of the event.")
    task_id: str | None = Field(None, description="The task ID of the event.")

    def __str__(self):
        string = dedent(f"""<chat_event>
            ID: {self.id}
            Step Number: {self.step_number}
            Event Type: {self.event_type}
            Timestamp: {self.timestamp}
            Agent Name: {self.agent_name}
            Session ID: {self.session_id}
            Task ID: {self.task_id}
            Data: {json.dumps(self.data)}
            </chat_event>""")
        return string

    def __repr__(self):
        return self.__str__()


class Importance(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Insight(BaseModel):
    id: str = Field(description="The unique identifier for the insight.")
    content: str = Field(description="The insight content")
    importance: Importance = Field(description="Importance level")
    source_event_id: str = Field(
        description="ID of the event that generated this insight"
    )
    tags: list[str] = Field(description="Tags for categorization")

    def __str__(self):
        string = dedent(f"""<insight>
            ID: {self.id}
            Content: {self.content}
            Importance: {self.importance}
            Source Event ID: {self.source_event_id}
            Tags: {self.tags}
            </insight>""")
        return string

    def __repr__(self):
        return self.__str__()


class Summary(BaseModel):
    id: str = Field(description="The unique identifier for the summary.")
    importance: Importance = Field(description="Importance level")
    content: str = Field(description="The summary content")

    def __str__(self):
        string = dedent(f"""<summary>
            ID: {self.id}
            Importance: {self.importance}
            Content: {self.content}
            </summary>""")
        return string

    def __repr__(self):
        return self.__str__()


class Memory(BaseModel):
    """定义 `Memory`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="", description="The name of the memory system")
    description: str = Field(
        default="", description="The description of the memory system"
    )
    save_path: str | None = Field(
        default=None, description="Path to save/load memory JSON file"
    )
    require_grad: bool = Field(
        default=False, description="Whether the memory system requires gradients"
    )

    def __init__(self, **kwargs):
        """初始化实例。"""
        super().__init__(**kwargs)
        # 更新相关状态。
        if not self.name:
            import inflection

            self.name = inflection.underscore(self.__class__.__name__)
        # 更新相关状态。
        if not self.description and self.__class__.__doc__:
            self.description = self.__class__.__doc__.strip().split("\n")[0]

    def __str__(self):
        return f"Memory(name={self.name}, description={self.description})"

    def __repr__(self):
        return self.__str__()


class MemoryConfig(BaseModel):
    """定义 `MemoryConfig`，封装相关数据与行为。"""

    name: str = Field(description="The name of the memory system")
    description: str = Field(description="The description of the memory system")
    require_grad: bool = Field(
        default=False, description="Whether the memory system requires gradients"
    )
    version: str = Field(default="1.0.0", description="Version of the memory system")

    cls: type[Memory] | None = Field(
        default=None, description="The class of the memory system"
    )
    instance: Any | None = Field(
        default=None, description="The instance of the memory system"
    )
    config: dict[str, Any] | None = Field(
        default_factory=dict,
        description="The initialization configuration of the memory system",
    )
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="The metadata of the memory system"
    )
    code: str | None = Field(
        default=None,
        description="Source code for dynamically generated memory classes (used when cls cannot be imported from a module)",
    )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """实现 `model_dump` 的业务逻辑。"""
        return {
            "name": self.name,
            "description": self.description,
            "require_grad": self.require_grad,
            "version": self.version,
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "config": self.config,
            "instance": None,  # 转换并规范化数据。
            "metadata": self.metadata,
            "code": self.code,
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "MemoryConfig":
        """实现 `model_validate` 的业务逻辑。"""
        name = data.get("name")
        description = data.get("description")
        require_grad = data.get("require_grad", False)  # 说明相关实现细节。
        version = data.get("version", "1.0.0")

        cls_ = None
        code = data.get("code")
        if code:
            class_name = dynamic_manager.extract_class_name_from_code(code)
            if class_name:
                try:
                    cls_ = dynamic_manager.load_class(
                        code, class_name=class_name, base_class=Memory, context="memory"
                    )
                except Exception:
                    cls_ = None
            else:
                cls_ = None
        else:
            cls_ = None

        config = data.get("config", {})
        instance = data.get("instance", None)
        metadata = data.get("metadata", {})

        return cls(
            name=name,
            description=description,
            require_grad=require_grad,
            version=version,
            cls=cls_,
            config=config,
            instance=instance,
            metadata=metadata,
            code=code,
        )

    def __str__(self):
        return f"MemoryConfig(name={self.name}, description={self.description}, version={self.version})"

    def __repr__(self):
        return self.__str__()
