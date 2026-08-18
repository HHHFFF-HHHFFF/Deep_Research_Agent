"""提供数据类型相关实现。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.dynamic import dynamic_manager
from src.environment.server import ecp
from src.logger import logger
from src.memory import EventType, memory_manager
from src.message.types import HumanMessage, Message, SystemMessage
from src.model import model_manager
from src.prompt import prompt_manager
from src.session import SessionContext
from src.skill.server import scp
from src.tool.server import tcp
from src.utils import (
    dedent,
    get_file_info,
)


class InputArgs(BaseModel):
    task: str = Field(description="The task to complete.")
    files: list[str] | None = Field(
        default=None, description="The files to attach to the task."
    )


class ACPErrorCode(Enum):
    """定义 `ACPErrorCode`，封装相关数据与行为。"""

    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    AGENT_NOT_FOUND = -32001


class ACPError(BaseModel):
    """定义 `ACPError`，封装相关数据与行为。"""

    code: ACPErrorCode
    message: str
    data: dict[str, Any] | None = None


class ACPRequest(BaseModel):
    """定义 `ACPRequest`，封装相关数据与行为。"""

    id: str | int = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: dict[str, Any] | None = None


class ACPResponse(BaseModel):
    """定义 `ACPResponse`，封装相关数据与行为。"""

    id: str | int
    result: dict[str, Any] | None = None
    error: ACPError | None = None


class AgentConfig(BaseModel):
    """定义 `AgentConfig`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the agent")
    description: str = Field(description="The description of the agent")
    version: str = Field(default="1.0.0", description="Version of the agent")
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    require_grad: bool = Field(
        default=False, description="Whether the agent requires gradients"
    )

    cls: Any | None = None
    config: dict[str, Any] | None = Field(
        default_factory=dict,
        description="The initialization configuration of the agent",
    )
    instance: Any | None = None

    code: str | None = Field(
        default=None,
        description="Source code for dynamically generated agent classes (used when cls cannot be imported from a module)",
    )

    function_calling: dict[str, Any] | None = Field(
        default=None, description="Default function calling representation"
    )
    text: str | None = Field(
        default=None, description="Default text representation of the agent"
    )
    args_schema: type[BaseModel] | None = Field(
        default=None, description="Default args schema (BaseModel type)"
    )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """实现 `model_dump` 的业务逻辑。"""

        result = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "version": self.version,
            "require_grad": self.require_grad,
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "config": self.config,
            "instance": None,
            "code": self.code,
            "function_calling": self.function_calling,
            "text": self.text,
            "args_schema": dynamic_manager.serialize_args_schema(self.args_schema)
            if self.args_schema
            else None,
        }

        return result

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> AgentConfig:
        """实现 `model_validate` 的业务逻辑。"""
        name = data.get("name")
        description = data.get("description")
        metadata = data.get("metadata", {})
        version = data.get("version")
        require_grad = data.get("require_grad", False)

        cls_ = None
        code = data.get("code")
        if code:
            class_name = dynamic_manager.extract_class_name_from_code(code)
            if class_name:
                try:
                    cls_ = dynamic_manager.load_class(
                        code, class_name=class_name, base_class=Agent, context="agent"
                    )
                except Exception:
                    cls_ = None
            else:
                cls_ = None
        else:
            cls_ = None

        config = data.get("config", {})
        instance = data.get("instance", None)

        function_calling = data.get("function_calling")
        text = data.get("text")
        args_schema = dynamic_manager.deserialize_args_schema(data.get("args_schema"))

        return cls(
            name=name,
            description=description,
            metadata=metadata,
            version=version,
            require_grad=require_grad,
            cls=cls_,
            config=config,
            instance=instance,
            function_calling=function_calling,
            text=text,
            args_schema=args_schema,
        )

    def __str__(self) -> str:
        return (
            f"AgentConfig(name={self.name}, "
            f"description={self.description}, "
            f"require_grad={self.require_grad})"
        )

    def __repr__(self) -> str:
        return self.__str__()


def format_actions(actions: list[BaseModel]) -> str:
    """格式化与 `format_actions` 对应的数据或状态。"""
    rows = []
    for action in actions:
        if isinstance(action.args, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in action.args.items())
        else:
            args_str = str(action.args)

        rows.append(
            {
                "Type": action.type if hasattr(action, "type") else "tool",
                "Name": action.name,
                "Args": args_str,
                "Output": action.output
                if hasattr(action, "output") and action.output is not None
                else None,
            }
        )

    df = pd.DataFrame(rows)

    if df["Output"].isna().all():
        df = df.drop(columns=["Output"])
    else:
        df["Output"] = df["Output"].fillna("None")

    return df.to_markdown(index=True)


class ActionInputArgs(BaseModel):
    type: str = Field(
        default="tool", description='The type of this action: "tool" or "skill".'
    )
    name: str = Field(description="The name of the tool or skill.")
    args: str = Field(
        description='The arguments as a JSON string. Must be a valid JSON object string. e.g., "{"result": "D", "reasoning": "Step 1: ..."}"'
    )


class ThinkOutput(BaseModel):
    thinking: str = Field(description="A structured <think>-style reasoning block.")
    evaluation_previous_goal: str = Field(
        description="One-sentence analysis of your last action."
    )
    memory: str = Field(description="1-3 sentences of specific memory.")
    next_goal: str = Field(description="State the next immediate goals and actions.")
    actions: list[ActionInputArgs] = Field(
        description=(
            "The list of actions (tool or skill calls) to execute in sequence. "
            'Each action has a "type" ("tool" or "skill"), a "name", and "args" (JSON string). '
            'e.g., [{"type": "tool", "name": "done", "args": "{"result": "D"}"}, '
            '{"type": "skill", "name": "hello-world", "args": "{"name": "Alice"}"}]'
        )
    )

    def __str__(self) -> str:
        return (
            f"Thinking: {self.thinking}\n"
            f"Evaluation of Previous Goal: {self.evaluation_previous_goal}\n"
            f"Memory: {self.memory}\n"
            f"Next Goal: {self.next_goal}\n"
            f"Actions:\n{format_actions(self.actions)}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()


class Agent(BaseModel):
    """定义 `Agent`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the agent.")
    description: str = Field(description="The description of the agent.")
    metadata: dict[str, Any] = Field(description="The metadata of the agent.")
    version: str = Field(default="1.0.0", description="Version of the agent")
    require_grad: bool = Field(
        default=False, description="Whether the agent requires gradients"
    )

    def __init__(
        self,
        workdir: str,
        name: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_name: str | None = None,
        prompt_name: str | None = None,
        memory_name: str | None = None,
        max_tools: int = 10,
        max_steps: int = 20,
        review_steps: int = 5,
        require_grad: bool = False,
        use_memory: bool = True,
        use_todo: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # 更新相关状态。
        self.name = name or self.name
        self.description = description or self.description
        self.metadata = metadata or self.metadata
        self.require_grad = require_grad

        # 更新相关状态。
        self.workdir = workdir

        # 更新相关状态。
        self.prompt_name = prompt_name
        self.memory_name = memory_name
        self.use_memory = use_memory
        self.model_name = model_name

        # 初始化相关状态。
        self.max_steps = max_steps if max_steps > 0 else int(1e8)
        self.max_tools = max_tools

        self.review_steps = review_steps
        self.use_todo = use_todo

    async def initialize(self) -> None:
        """初始化组件及其依赖资源。"""
        logger.info(f"| 📁 Agent working directory: {self.workdir}")

    def __str__(self) -> str:
        return f"Agent(name={self.name}, model={self.model_name}, prompt_name={self.prompt_name})"

    def __repr__(self) -> str:
        return self.__str__()

    async def _extract_file_content(self, file: str) -> dict[str, Any]:
        """实现 `_extract_file_content` 的业务逻辑。"""

        info = get_file_info(file)

        # 处理文件与路径。
        input_payload = {
            "name": "mdify",
            "input": {
                "file_path": file,
                "output_format": "markdown",
            },
        }
        tool_response = await tcp(**input_payload)
        file_content = tool_response.message

        # 处理模型调用。
        system_prompt = "You are a helpful assistant that summarizes file content."

        user_prompt = dedent(
            f"""
            Summarize the following file content as 1-3 sentences:
            {file_content}
        """
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        model_response = await model_manager(model=self.model_name, messages=messages)

        info["content"] = file_content
        info["summary"] = model_response.message

        return info

    async def _generate_enhanced_task(
        self, task: str, files: list[dict[str, Any]]
    ) -> str:
        """实现 `_generate_enhanced_task` 的业务逻辑。"""

        attach_files_string = "\n".join(
            [f"File: {file['path']}\nSummary: {file['summary']}" for file in files]
        )

        enhanced_task = dedent(
            f"""
            - Task:
            {task}
            - Attach files:
            {attach_files_string}
        """
        )
        return enhanced_task

    async def _get_agent_context(
        self, task: str, step_number: int = 0, ctx: SessionContext = None, **kwargs
    ) -> dict[str, Any]:
        """实现 `_get_agent_context` 的业务逻辑。"""
        task = f"<task>{task}</task>"

        step_info_description = (
            f"Step {step_number + 1} of {self.max_steps} max possible steps\n"
        )
        time_str = datetime.now(timezone.utc).isoformat()
        step_info_description += f"Current date and time: {time_str}"
        step_info = dedent(f"""
            <step_info>
            {step_info_description}
            </step_info>
        """)

        # 处理记忆或缓存状态。
        memory = ""
        if self.use_memory and self.memory_name:
            state = await memory_manager.get_state(
                name=self.memory_name, n=self.review_steps, ctx=ctx
            )
            events = state["events"]
            summaries = state["summaries"]
            insights = state["insights"]

            # 处理版本与历史记录。
            memory += "<agent_history>"
            for event in events:
                memory += f"<step_{event.step_number}>\n"
                if event.event_type == EventType.TASK_START:
                    memory += f"Task Start: {event.data.get('task', event.data.get('message', ''))}\n"
                elif event.event_type == EventType.TASK_END:
                    memory += f"Task End: {event.data.get('result', '')}\n"
                elif event.event_type == EventType.TOOL_STEP:
                    memory += f"Evaluation of Previous Step: {event.data.get('evaluation_previous_goal', '')}\n"
                    memory += f"Memory: {event.data.get('memory', '')}\n"
                    memory += f"Next Goal: {event.data.get('next_goal', '')}\n"
                    memory += f"Action Results: {event.data.get('actions', event.data.get('tool', ''))}\n"
                memory += "\n"
                memory += f"</step_{event.step_number}>\n"
            memory += "</agent_history>"

            # 处理记忆或缓存状态。
            memory += "<memory>"
            if len(summaries) > 0:
                memory += dedent(
                    f"""
                    <summaries>
                    {chr(10).join([str(summary) for summary in summaries])}
                    </summaries>
                """
                )
            else:
                memory += "<summaries>[Current summaries are empty.]</summaries>\n"
            if len(insights) > 0:
                memory += dedent(
                    f"""
                    <insights>
                    {chr(10).join([str(insight) for insight in insights])}
                    </insights>
                """
                )
            else:
                memory += "<insights>[Current insights are empty.]</insights>\n"
            memory += "</memory>"

        else:
            memory += "<agent_history>[Agent history is disabled.]</agent_history>\n"
            memory += "<memory>[Memory is disabled.]</memory>\n"

        if self.use_todo:
            todo = "<todo>"
            todo_tool = await tcp.get("todo")
            todo_contents = todo_tool.get_todo_content(ctx=ctx)
            todo += todo_contents
            todo += "</todo>"
        else:
            todo = "<todo>[Todo is disabled.]</todo>\n"

        agent_context = dedent(f"""
            <agent_context>
            {task}
            {step_info}
            {memory}
            {todo}
            </agent_context>
        """)

        return {
            "agent_context": agent_context,
        }

    async def _get_environment_context(
        self, ctx: SessionContext, **kwargs
    ) -> dict[str, Any]:
        """实现 `_get_environment_context` 的业务逻辑。"""

        environment_context = "<environment_context>"
        # 配置相关参数。
        for env_name in config.env_names:
            env_info = await ecp.get_info(env_name)
            rule_string = env_info.rules
            rule_string = dedent(f"""
                <rules>
                {rule_string}
                </rules>
            """)

            env_state = await ecp.get_state(env_name, ctx=ctx)
            state_string = "<state>"
            state_string += env_state["state"]
            extra = env_state["extra"]

            if "screenshots" in extra:
                for screenshot in extra["screenshots"]:
                    state_string += (
                        f"\n<img src={screenshot.screenshot_path} "
                        f"alt={screenshot.screenshot_description}/>"
                    )
            state_string += "</state>"

            environment_context += dedent(f"""
                <{env_name}>
                {rule_string}
                {state_string}
                </{env_name}>
            """)

        environment_context += "</environment_context>"
        return {
            "environment_context": environment_context,
        }

    async def _get_tool_context(self, ctx: SessionContext, **kwargs) -> dict[str, Any]:
        """实现 `_get_tool_context` 的业务逻辑。"""
        tool_context = "<tool_context>"

        tool_context += dedent(f"""
            <available_tools>
            {await tcp.get_contract()}
            </available_tools>
        """)

        tool_context += "</tool_context>"
        return {
            "tool_context": tool_context,
        }

    async def _get_skill_context(self, ctx: SessionContext, **kwargs) -> dict[str, Any]:
        """实现 `_get_skill_context` 的业务逻辑。"""
        skill_content = await scp.get_context()
        if not skill_content:
            skill_context = "<skill_context>[No skills loaded.]</skill_context>\n"
        else:
            skill_context = f"<skill_context>\n{skill_content}\n</skill_context>"
        return {
            "skill_context": skill_context,
        }

    async def _get_messages(
        self, task: str, ctx: SessionContext, **kwargs
    ) -> list[Message]:
        """实现 `_get_messages` 的业务逻辑。"""

        system_modules = {"max_tools": self.max_tools, "workdir": self.workdir}
        agent_message_modules = {"task": task}

        agent_message_modules.update(await self._get_agent_context(task, ctx=ctx))
        agent_message_modules.update(await self._get_environment_context(ctx=ctx))
        agent_message_modules.update(await self._get_tool_context(ctx=ctx))
        agent_message_modules.update(await self._get_skill_context(ctx=ctx))

        messages = await prompt_manager.get_messages(
            prompt_name=self.prompt_name,
            system_modules=system_modules,
            agent_modules=agent_message_modules,
        )

        return messages

    async def __call__(
        self,
        task: str,
        files: list[str] | None = None,
        ctx: SessionContext | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """执行组件调用并返回结果。"""
        raise NotImplementedError(
            "__all__ method is not implemented by the child class"
        )


class AgentExtra(BaseModel):
    """定义 `AgentExtra`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    file_path: str | list[str] | None = Field(
        default=None, description="The file path of the extra data"
    )
    data: dict[str, Any] | None = Field(
        default=None, description="The data of the extra data"
    )
    parsed_model: BaseModel | None = Field(
        default=None, description="The parsed model of the extra data"
    )


class AgentResponse(BaseModel):
    """定义 `AgentResponse`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    success: bool = Field(description="Whether the agent has completed the task.")
    message: str = Field(description="The message of the agent.")
    extra: AgentExtra | None = Field(
        default=None, description="The extra data of the agent."
    )


__all__ = [
    "ACPError",
    "ACPErrorCode",
    "ACPRequest",
    "ACPResponse",
    "ActionInputArgs",
    "Agent",
    "AgentConfig",
    "AgentResponse",
    "InputArgs",
    "ThinkOutput",
]
