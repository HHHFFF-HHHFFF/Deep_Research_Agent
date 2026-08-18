"""提供tool calling agent相关实现。"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse, ThinkOutput
from src.config import config
from src.environment.server import ecp
from src.logger import logger
from src.memory import EventType, memory_manager
from src.model import model_manager
from src.registry import AGENT
from src.session import SessionContext
from src.skill.server import scp
from src.tool.server import tcp
from src.tracer import Record, Tracer
from src.utils import dedent, parse_tool_args


@AGENT.register_module(force=True)
class ToolCallingAgent(Agent):
    """定义 `ToolCallingAgent`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(
        default="tool_calling", description="The name of the tool calling agent."
    )
    description: str = Field(
        default="A tool calling agent that can call tools to complete tasks.",
        description="The description of the tool calling agent.",
    )
    metadata: dict[str, Any] = Field(
        default={}, description="The metadata of the tool calling agent."
    )
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
        **kwargs,
    ):
        # 更新相关状态。
        if not prompt_name:
            prompt_name = "tool_calling"

        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_name=memory_name,
            max_tools=max_tools,
            max_steps=max_steps,
            review_steps=review_steps,
            require_grad=require_grad,
            **kwargs,
        )

    async def initialize(self):
        """初始化组件及其依赖资源。"""
        self.tracer_save_path = os.path.join(self.workdir, "tracer.json")
        await super().initialize()

    async def _get_tracer_and_record(self) -> tuple[Tracer, Record]:
        """实现 `_get_tracer_and_record` 的业务逻辑。"""
        tracer = Tracer()
        record = Record()

        if os.path.exists(self.tracer_save_path):
            await tracer.load_from_json(self.tracer_save_path)
            last_record = await tracer.get_last_record()
            if last_record:
                record = last_record

        return tracer, record

    async def _get_environment_context(
        self, ctx: SessionContext, record: Record = None, **kwargs
    ) -> dict[str, Any]:
        """实现 `_get_environment_context` 的业务逻辑。"""

        environment_context = "<environment_context>"
        record_observation = {}

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
            record_observation[env_name] = extra

            if "screenshots" in extra:
                for screenshot in extra["screenshots"]:
                    state_string += f"\n<img src={screenshot.screenshot_path} alt={screenshot.screenshot_description}/>"
            state_string += "</state>"

            environment_context += dedent(f"""
                <{env_name}>
                {rule_string}
                {state_string}
                </{env_name}>
            """)

        if record is not None:
            record.observation = record_observation

        environment_context += "</environment_context>"
        return {
            "environment_context": environment_context,
        }

    async def _get_tool_context(
        self, ctx: SessionContext, record: Record = None, **kwargs
    ) -> dict[str, Any]:
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

    async def _think_and_tool(
        self,
        messages: list[BaseMessage],
        task_id: str,
        step_number: int,
        record: Record = None,
        ctx: SessionContext = None,
        **kwargs,
    ) -> dict[str, Any]:
        """实现 `_think_and_tool` 的业务逻辑。"""

        done = False
        result = None
        reasoning = None

        record_data = {
            "thinking": None,
            "evaluation_previous_goal": None,
            "memory": None,
            "next_goal": None,
            "actions": [],
        }

        try:
            think_output = await model_manager(
                model=self.model_name, messages=messages, response_format=ThinkOutput
            )
            think_output = think_output.extra.parsed_model

            thinking = think_output.thinking
            evaluation_previous_goal = think_output.evaluation_previous_goal
            memory = think_output.memory
            next_goal = think_output.next_goal
            actions = think_output.actions

            record_data["thinking"] = thinking
            record_data["evaluation_previous_goal"] = evaluation_previous_goal
            record_data["memory"] = memory
            record_data["next_goal"] = next_goal

            logger.info(f"| 💭 Thinking: {thinking}")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 🔧 Actions to execute: {actions}")

            # 处理工具调用。
            action_results = []

            for i, action in enumerate(actions):
                action_type = action.type
                action_name = action.name
                action_args_str = action.args
                action_args = (
                    parse_tool_args(action_args_str) if action_args_str else {}
                )

                logger.info(
                    f"| 📝 Action {i + 1}/{len(actions)}: [{action_type}] {action_name}"
                )
                logger.info(f"| 📝 Args: {action_args}")

                if action_type == "skill":
                    # 说明相关实现细节。
                    response = await scp(
                        name=action_name,
                        input=action_args,
                        ctx=ctx,
                    )
                    action_result = response.message
                    action_extra = (
                        response.extra if hasattr(response, "extra") else None
                    )

                    logger.info(
                        f"| ✅ Skill '{action_name}' completed (success={response.success})"
                    )
                    logger.info(f"| 📄 Result: {str(action_result)[:500]}")

                    action_dict = action.model_dump()
                    action_dict["output"] = action_result
                    action_results.append(action_dict)

                    record_extra = {}
                    record_extra.update(action_dict)
                    if action_extra is not None:
                        record_extra["extra"] = action_extra.model_dump()
                    record_data["actions"].append(record_extra)

                else:
                    # 处理工具调用。
                    tool_response = await tcp(
                        name=action_name,
                        input=action_args,
                        ctx=ctx,
                    )
                    action_result = tool_response.message
                    action_extra = (
                        tool_response.extra if hasattr(tool_response, "extra") else None
                    )

                    logger.info(f"| ✅ Tool '{action_name}' completed")
                    logger.info(f"| 📄 Result: {action_result!s}")

                    action_dict = action.model_dump()
                    action_dict["output"] = action_result
                    action_results.append(action_dict)

                    record_extra = {}
                    record_extra.update(action_dict)
                    if action_extra is not None:
                        record_extra["extra"] = action_extra.model_dump()
                    record_data["actions"].append(record_extra)

                    if action_name == "done":
                        done = True
                        result = action_result
                        reasoning = (
                            action_extra.data.get("reasoning", None)
                            if action_extra and action_extra.data
                            else None
                        )
                        break

            event_data = {
                "thinking": thinking,
                "evaluation_previous_goal": evaluation_previous_goal,
                "memory": memory,
                "next_goal": next_goal,
                "actions": action_results,
            }

            if record is not None:
                record.tool = record_data

            # 处理记忆或缓存状态。
            memory_name = self.memory_name

            # 处理记忆或缓存状态。
            if self.use_memory and memory_name:
                await memory_manager.add_event(
                    memory_name=memory_name,
                    step_number=step_number,
                    event_type=EventType.TOOL_STEP,
                    data=event_data,
                    agent_name=self.name,
                    task_id=task_id,
                    ctx=ctx,
                )

        except Exception as e:
            logger.error(f"| Error in thinking and tool step: {e}")

        response_dict = {"done": done, "result": result, "reasoning": reasoning}
        return response_dict

    async def __call__(
        self, task: str, files: list[str] | None = None, **kwargs
    ) -> AgentResponse:
        """执行组件调用并返回结果。"""
        logger.info(f"| 🚀 Starting ToolCallingAgent: {task}")

        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()

        # 创建所需对象。
        tracer, record = await self._get_tracer_and_record()

        if files:
            logger.info(f"| 📂 Attached files: {files}")
            files = await asyncio.gather(
                *[self._extract_file_content(file) for file in files]
            )
            enhanced_task = await self._generate_enhanced_task(task, files, ctx=ctx)
        else:
            enhanced_task = task

        # 处理记忆或缓存状态。
        memory_name = self.memory_name

        task_id = "task_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        logger.info(f"| 📝 Context ID: {ctx.id}, Task ID: {task_id}")

        # 处理记忆或缓存状态。
        if self.use_memory and memory_name:
            await memory_manager.start_session(memory_name=memory_name, ctx=ctx)

            # 执行异步任务。
            await memory_manager.add_event(
                memory_name=memory_name,
                step_number=0,
                event_type=EventType.TASK_START,
                data={"task": enhanced_task},
                agent_name=self.name,
                task_id=task_id,
                ctx=ctx,
            )
        else:
            logger.info(
                f"| ⏭️ Memory disabled (use_memory={self.use_memory}), skipping session management"
            )

        # 初始化相关状态。
        messages = await self._get_messages(enhanced_task, ctx=ctx)

        # 说明相关实现细节。
        step_number = 0

        while step_number < self.max_steps:
            logger.info(f"| 🔄 Step {step_number + 1}/{self.max_steps}")

            # 说明相关实现细节。
            response = await self._think_and_tool(
                messages, task_id, step_number, ctx=ctx, record=record
            )
            step_number += 1

            # 持久化相关数据。
            await tracer.add_record(
                observation=record.observation,
                tool=record.tool,
                task_id=task_id,
                ctx=ctx,
            )
            await tracer.save_to_json(self.tracer_save_path)

            # 持久化相关数据。
            messages = await self._get_messages(enhanced_task, ctx=ctx)

            if response["done"]:
                break

        # 说明相关实现细节。
        if step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping...")
            response = {
                "done": False,
                "result": "The task has not been completed.",
                "reasoning": "Reached the maximum number of steps.",
            }

        # 处理记忆或缓存状态。
        memory_name = self.memory_name

        # 处理记忆或缓存状态。
        if self.use_memory and memory_name:
            await memory_manager.add_event(
                memory_name=memory_name,
                step_number=step_number,
                event_type=EventType.TASK_END,
                data=response,
                agent_name=self.name,
                task_id=task_id,
                ctx=ctx,
            )

            # 持久化相关数据。
            await memory_manager.end_session(memory_name=memory_name, ctx=ctx)

        # 持久化相关数据。
        await tracer.save_to_json(self.tracer_save_path)

        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")

        return AgentResponse(
            success=response["done"],
            message=response["result"],
            extra=AgentExtra(data=response),
        )
