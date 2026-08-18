"""提供todo相关实现。"""

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.registry import TOOL
from src.session import SessionContext
from src.tool.types import Tool, ToolExtra, ToolResponse
from src.utils import (
    assemble_project_path,
    file_lock,
    read_json_file,
    write_json_file,
    write_text_file,
)


class Step(BaseModel):
    """定义 `Step`，封装相关数据与行为。"""

    id: str = Field(description="Unique step ID")
    name: str = Field(description="Step name/description")
    parameters: dict[str, Any] | None = Field(
        default=None, description="Step parameters"
    )
    status: str = Field(
        default="pending", description="Step status: pending, success, failed"
    )
    result: str | None = Field(default=None, description="Step result (1-3 sentences)")
    priority: str = Field(
        default="medium", description="Step priority: high, medium, low"
    )
    category: str | None = Field(default=None, description="Step category")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str | None = Field(default=None, description="Last update timestamp")


class Todo(BaseModel):
    """定义 `Todo`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: str = Field(description="Unique identifier for this todo instance")
    todo_file: str = Field(description="Path to the todo.md file")
    steps_file: str = Field(description="Path to the steps JSON file")
    steps: list[Step] = Field(default_factory=list, description="List of steps")

    def __init__(
        self,
        id: str,
        todo_file: str,
        steps_file: str,
        steps: list[Step] | None = None,
        **kwargs,
    ):
        super().__init__(id=id, todo_file=todo_file, steps_file=steps_file, **kwargs)
        if steps is not None:
            self.steps = steps

    def _generate_step_id(self) -> str:
        """实现 `_generate_step_id` 的业务逻辑。"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}"

    async def _save_steps(self) -> None:
        """实现 `_save_steps` 的业务逻辑。"""
        try:
            async with file_lock(self.steps_file):
                await write_json_file(
                    self.steps_file,
                    [step.model_dump() for step in self.steps],
                    indent=2,
                )
        except Exception as e:
            logger.error(f"| ❌ Error saving steps: {e}")

    async def _sync_to_markdown(self) -> None:
        """实现 `_sync_to_markdown` 的业务逻辑。"""
        try:
            async with file_lock(self.todo_file):
                content = "# Todo List\n\n"

                for step in self.steps:
                    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                        step.priority, "🟡"
                    )

                    status_emoji = {
                        "pending": "⏳",
                        "success": "✅",
                        "failed": "❌",
                    }.get(step.status, "⏳")

                    category_text = f" [{step.category}]" if step.category else ""

                    # 创建所需对象。
                    if step.status == "pending":
                        checkbox = "[ ]"
                    else:
                        checkbox = "[x]"

                    step_line = f"- {checkbox} **{step.id}** {priority_emoji} {status_emoji} {step.name}{category_text}"

                    if step.parameters:
                        step_line += f" *(params: {json.dumps(step.parameters)})*"

                    step_line += f" *(created: {step.created_at}*"

                    if step.updated_at:
                        step_line += f", updated: {step.updated_at}"

                    if step.result:
                        step_line += f", result: {step.result}"

                    step_line += ")"
                    content += step_line + "\n"

                await write_text_file(self.todo_file, content)
        except Exception as e:
            logger.error(f"| ❌ Error syncing to markdown: {e}")

    async def add_step(
        self,
        task: str,
        priority: str = "medium",
        category: str | None = None,
        parameters: dict[str, Any] | None = None,
        after_step_id: str | None = None,
        step_id: str | None = None,
    ) -> Step:
        """添加与 `add_step` 对应的数据或状态。"""
        if not task:
            raise ValueError("Step description is required")

        # 说明相关实现细节。
        if step_id is None:
            step_id = self._generate_step_id()
        else:
            # 加载所需数据。
            for step in self.steps:
                if step.id == step_id:
                    raise ValueError(f"Step ID {step_id} already exists")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        new_step = Step(
            id=step_id,
            name=task,
            parameters=parameters,
            status="pending",
            priority=priority,
            category=category,
            created_at=timestamp,
        )

        # 说明相关实现细节。
        if after_step_id:
            # 说明相关实现细节。
            insert_index = -1
            for i, step in enumerate(self.steps):
                if step.id == after_step_id:
                    insert_index = i + 1
                    break

            # 说明相关实现细节。
            if insert_index == -1:
                try:
                    index = int(after_step_id)
                    if 0 <= index < len(self.steps):
                        insert_index = index + 1
                except ValueError:
                    pass

            if insert_index == -1:
                # 说明相关实现细节。
                self.steps.append(new_step)
            else:
                # 说明相关实现细节。
                self.steps.insert(insert_index, new_step)
        else:
            # 说明相关实现细节。
            self.steps.append(new_step)

        # 持久化相关数据。
        await self._save_steps()
        await self._sync_to_markdown()

        return new_step

    async def complete_step(
        self, step_id: str, status: str, result: str | None = None
    ) -> Step:
        """实现 `complete_step` 的业务逻辑。"""
        if not step_id:
            raise ValueError("Step ID is required")

        if status not in ["success", "failed"]:
            raise ValueError("Status must be 'success' or 'failed'")

        # 说明相关实现细节。
        step = None
        for s in self.steps:
            if s.id == step_id:
                step = s
                break

        if not step:
            raise ValueError(f"Step {step_id} not found")

        if step.status != "pending":
            raise ValueError(
                f"Step {step_id} is already completed with status: {step.status}"
            )

        # 更新相关状态。
        step.status = status
        step.result = result
        step.updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        # 持久化相关数据。
        await self._save_steps()
        await self._sync_to_markdown()

        return step

    async def update_step(
        self,
        step_id: str,
        task: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Step:
        """更新与 `update_step` 对应的数据或状态。"""
        if not step_id:
            raise ValueError("Step ID is required")

        # 说明相关实现细节。
        step = None
        for s in self.steps:
            if s.id == step_id:
                step = s
                break

        if not step:
            raise ValueError(f"Step {step_id} not found")

        # 更新相关状态。
        if task:
            step.name = task
        if parameters is not None:
            step.parameters = parameters

        step.updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        # 持久化相关数据。
        await self._save_steps()
        await self._sync_to_markdown()

        return step

    async def clear_completed(self) -> list[Step]:
        """实现 `clear_completed` 的业务逻辑。"""
        completed_steps = [
            step for step in self.steps if step.status in ["success", "failed"]
        ]

        # 移除相关数据或组件。
        self.steps = [step for step in self.steps if step.status == "pending"]

        # 持久化相关数据。
        await self._save_steps()
        await self._sync_to_markdown()

        return completed_steps

    def get_content(self) -> str:
        """获取与 `get_content` 对应的数据或状态。"""
        if not os.path.exists(self.todo_file):
            return "[Current todo.md is empty, fill it with your plan when applicable]"

        try:
            with open(self.todo_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return "[Current todo.md is empty, fill it with your plan when applicable]"


_TODO_TOOL_DESCRIPTION = """Todo tool for managing a todo.md file with task decomposition and step tracking.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

Available `action` parameters:
1. add: Add a new step to the todo list at the end or after a specific step.
    - task: The description of the step.
    - priority: The priority of the step.
    - category: The category of the step.
    - parameters: Optional parameters for the step.
    - after_step_id: Optional step ID to insert after (if not provided, adds to end).
2. complete: Mark step as completed (success or failed).
    - step_id: The ID of the step to complete.
    - status: Completion status: "success" or "failed".
    - result: Result description (1-3 sentences).
3. update: Update step information.
    - step_id: The ID of the step to update.
    - task: New step description.
    - parameters: New step parameters.
4. list: List all steps with their status.
5. clear: Clear completed steps.
6. show: Show the complete todo.md file content.
7. export: Export todo.md to a specified path.
    - export_path: The target path to export the todo.md file.
8. cleanup: Clean up and remove the todo from cache (call when done with the todo list).

Example: {"name": "todo", "args": {"action": "add", "task": "Task description", "priority": "high", "category": "work"}}
Example: {"name": "todo", "args": {"action": "complete", "step_id": "step_1", "status": "success", "result": "Completed successfully"}}

The todo.md file is maintained in the base directory and follows a structured format for task management.
"""


@TOOL.register_module(force=True)
class TodoTool(Tool):
    """定义 `TodoTool`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "todo"
    description: str = _TODO_TOOL_DESCRIPTION
    metadata: dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(
        default=False, description="Whether the tool requires gradients"
    )

    # 配置相关参数。
    base_dir: str = Field(
        default="workdir/todo", description="The base directory for saving todo files."
    )

    def __init__(
        self, base_dir: str | None = None, require_grad: bool = False, **kwargs
    ):
        """初始化实例。"""
        super().__init__(require_grad=require_grad, **kwargs)

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(self.base_dir)

        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)

        # 处理记忆或缓存状态。
        # 待办：后续完善此处实现。
        self._todo_cache: dict[str, Todo] = {}
        # 待办：后续完善此处实现。
        self._todo_locks: dict[str, asyncio.Lock] = {}
        # 处理记忆或缓存状态。
        self._cache_lock = asyncio.Lock()

    def _get_todo_file_path(self, id: str) -> str:
        """实现 `_get_todo_file_path` 的业务逻辑。"""
        safe_id = re.sub(r"[^\w\s-]", "", id).strip().replace(" ", "_")
        if not safe_id:
            safe_id = "todo"
        return os.path.join(self.base_dir, f"{safe_id}_todo.md")

    def _get_steps_file_path(self, id: str) -> str:
        """实现 `_get_steps_file_path` 的业务逻辑。"""
        safe_id = re.sub(r"[^\w\s-]", "", id).strip().replace(" ", "_")
        if not safe_id:
            safe_id = "todo"
        return os.path.join(self.base_dir, f"{safe_id}_steps.json")

    async def _get_or_create_todo(self, id: str) -> tuple[Todo, asyncio.Lock]:
        """实现 `_get_or_create_todo` 的业务逻辑。"""
        async with self._cache_lock:
            # 创建所需对象。
            if id not in self._todo_locks:
                self._todo_locks[id] = asyncio.Lock()

            # 待办：后续完善此处实现。
            if id not in self._todo_cache:
                todo_file = self._get_todo_file_path(id)
                steps_file = self._get_steps_file_path(id)

                # 加载所需数据。
                steps = []
                if os.path.exists(steps_file):
                    try:
                        data = await read_json_file(steps_file)
                        steps = [Step(**step_data) for step_data in data]
                    except Exception:
                        steps = []

                self._todo_cache[id] = Todo(
                    id=id, todo_file=todo_file, steps_file=steps_file, steps=steps
                )
                logger.info(
                    f"| 📝 Created new todo cache for id: {id} (file_path: {todo_file})"
                )
            else:
                logger.info(
                    f"| 📂 Using existing todo cache for id: {id} (steps: {len(self._todo_cache[id].steps)})"
                )

            return self._todo_cache[id], self._todo_locks[id]

    async def _cleanup_todo(self, id: str):
        """实现 `_cleanup_todo` 的业务逻辑。"""
        async with self._cache_lock:
            if id in self._todo_cache:
                del self._todo_cache[id]
                logger.info(f"| 🧹 Removed todo from cache: {id}")
            if id in self._todo_locks:
                del self._todo_locks[id]

    async def __call__(
        self,
        action: str,
        task: str | None = None,
        step_id: str | None = None,
        status: str | None = None,
        result: str | None = None,
        priority: str | None = "medium",
        category: str | None = None,
        parameters: dict[str, Any] | None = None,
        after_step_id: str | None = None,
        export_path: str | None = None,
        **kwargs,
    ) -> ToolResponse:
        """执行组件调用并返回结果。"""
        try:
            # 处理工具调用。
            ctx = kwargs.get("ctx")
            id = ctx.id

            # 清理并释放相关资源。
            if action == "cleanup":
                await self._cleanup_todo(id)
                return ToolResponse(
                    success=True, message=f"✅ Cleaned up todo cache for id: {id}"
                )

            # 待办：后续完善此处实现。
            todo, todo_lock = await self._get_or_create_todo(id)

            # 待办：后续完善此处实现。
            async with todo_lock:
                logger.info(
                    f"| 📝 TodoTool action: {action} (id: {id}, todo_file: {todo.todo_file}, steps: {len(todo.steps)})"
                )

                if action == "add":
                    return await self._handle_add(
                        todo,
                        task,
                        priority,
                        category,
                        parameters,
                        after_step_id,
                        step_id,
                    )
                if action == "complete":
                    return await self._handle_complete(todo, step_id, status, result)
                if action == "update":
                    return await self._handle_update(todo, step_id, task, parameters)
                if action == "list":
                    return self._handle_list(todo)
                if action == "clear":
                    return await self._handle_clear(todo)
                if action == "show":
                    return self._handle_show(todo)
                if action == "export":
                    return await self._handle_export(todo, export_path)

                return ToolResponse(
                    success=False,
                    message=(
                        f"Unknown action: {action}. "
                        "Available actions: add, complete, update, list, clear, show, export, cleanup"
                    ),
                )

        except Exception as e:
            logger.error(f"| ❌ Error in TodoTool: {e}")
            import traceback

            return ToolResponse(
                success=False,
                message=f"Error executing todo action '{action}': {e!s}\n{traceback.format_exc()}",
            )

    async def _handle_add(
        self,
        todo: Todo,
        task: str,
        priority: str,
        category: str | None,
        parameters: dict[str, Any] | None,
        after_step_id: str | None,
        step_id: str | None,
    ) -> ToolResponse:
        """实现 `_handle_add` 的业务逻辑。"""
        if not task:
            return ToolResponse(
                success=False,
                message="Error: Step description is required for add action",
            )

        try:
            new_step = await todo.add_step(
                task=task,
                priority=priority,
                category=category,
                parameters=parameters,
                after_step_id=after_step_id,
                step_id=step_id,
            )

            message = f"✅ Added step {new_step.id}: {task} (priority: {priority})"
            if after_step_id:
                message = f"✅ Added step {new_step.id} after {after_step_id}: {task} (priority: {priority})"

            logger.info(f"| {message}")
            return ToolResponse(
                success=True,
                message=message,
                extra=ToolExtra(
                    file_path=todo.todo_file,
                    data={
                        "step_id": new_step.id,
                        "after_step_id": after_step_id,
                        "task": task,
                        "priority": priority,
                        "category": category,
                        "parameters": parameters,
                    },
                ),
            )
        except ValueError as e:
            return ToolResponse(success=False, message=f"Error: {e!s}")

    async def _handle_complete(
        self, todo: Todo, step_id: str, status: str, result: str | None
    ) -> ToolResponse:
        """实现 `_handle_complete` 的业务逻辑。"""
        if not step_id:
            return ToolResponse(
                success=False, message="Error: Step ID is required for complete action"
            )

        if not status:
            return ToolResponse(
                success=False, message="Error: Status is required for complete action"
            )

        try:
            step = await todo.complete_step(
                step_id=step_id, status=status, result=result
            )

            return ToolResponse(
                success=True,
                message=f"✅ Completed step {step_id} with status: {status}",
                extra=ToolExtra(
                    file_path=todo.todo_file,
                    data={
                        "step_id": step_id,
                        "status": status,
                        "result": result,
                        "updated_at": step.updated_at,
                        "step_name": step.name,
                    },
                ),
            )
        except ValueError as e:
            return ToolResponse(success=False, message=f"Error: {e!s}")

    async def _handle_update(
        self,
        todo: Todo,
        step_id: str,
        task: str | None,
        parameters: dict[str, Any] | None,
    ) -> ToolResponse:
        """实现 `_handle_update` 的业务逻辑。"""
        if not step_id:
            return ToolResponse(
                success=False, message="Error: Step ID is required for update action"
            )

        try:
            step = await todo.update_step(
                step_id=step_id, task=task, parameters=parameters
            )

            return ToolResponse(
                success=True,
                message=f"✅ Updated step {step_id}",
                extra=ToolExtra(
                    file_path=todo.todo_file,
                    data={
                        "step_id": step_id,
                        "updated_fields": {
                            "task": task if task else None,
                            "parameters": parameters
                            if parameters is not None
                            else None,
                        },
                        "updated_at": step.updated_at,
                        "step_name": step.name,
                    },
                ),
            )
        except ValueError as e:
            return ToolResponse(success=False, message=f"Error: {e!s}")

    def _handle_list(self, todo: Todo) -> ToolResponse:
        """实现 `_handle_list` 的业务逻辑。"""
        if not todo.steps:
            return ToolResponse(
                success=False,
                message="No steps found. Use 'add' action to create your first step.",
            )

        result = "📋 Todo Steps:\n\n"
        for step in todo.steps:
            status_emoji = {"pending": "⏳", "success": "✅", "failed": "❌"}.get(
                step.status, "⏳"
            )

            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                step.priority, "🟡"
            )

            category_text = f" [{step.category}]" if step.category else ""
            result += f"**{step.id}** {priority_emoji} {status_emoji} {step.name}{category_text}\n"

            if step.parameters:
                result += f"  Parameters: {json.dumps(step.parameters)}\n"

            if step.result:
                result += f"  Result: {step.result}\n"

            result += f"  Created: {step.created_at}"
            if step.updated_at:
                result += f", Updated: {step.updated_at}"
            result += "\n\n"

        return ToolResponse(
            success=True,
            message=result,
            extra=ToolExtra(
                file_path=todo.todo_file,
                data={
                    "total_steps": len(todo.steps),
                    "steps": [step.model_dump() for step in todo.steps],
                    "pending_count": len(
                        [s for s in todo.steps if s.status == "pending"]
                    ),
                    "completed_count": len(
                        [s for s in todo.steps if s.status in ["success", "failed"]]
                    ),
                },
            ),
        )

    async def _handle_clear(self, todo: Todo) -> ToolResponse:
        """实现 `_handle_clear` 的业务逻辑。"""
        completed_steps = await todo.clear_completed()

        if not completed_steps:
            return ToolResponse(success=False, message="No completed steps to remove")

        return ToolResponse(
            success=True,
            message=f"✅ Removed {len(completed_steps)} completed step(s)",
            extra=ToolExtra(
                file_path=todo.todo_file,
                data={
                    "removed_count": len(completed_steps),
                    "removed_steps": [step.model_dump() for step in completed_steps],
                    "remaining_steps": len(todo.steps),
                },
            ),
        )

    def _handle_show(self, todo: Todo) -> ToolResponse:
        """实现 `_handle_show` 的业务逻辑。"""
        content = todo.get_content()

        if content.startswith("[Current todo.md is empty"):
            return ToolResponse(
                success=False,
                message="No todo file found. Use 'add' action to create your first step.",
            )

        return ToolResponse(
            success=True,
            message=f"📄 Todo.md content:\n\n```markdown\n{content}\n```",
            extra=ToolExtra(
                file_path=todo.todo_file,
                data={
                    "content": content,
                    "file_size": len(content),
                    "total_steps": len(todo.steps),
                },
            ),
        )

    async def _handle_export(self, todo: Todo, export_path: str) -> ToolResponse:
        """实现 `_handle_export` 的业务逻辑。"""
        if not export_path:
            return ToolResponse(
                success=False,
                message="Error: Export path is required for export action",
            )

        try:
            # 待办：后续完善此处实现。
            await todo._sync_to_markdown()

            content = todo.get_content()
            if content.startswith("[Current todo.md is empty"):
                return ToolResponse(
                    success=False,
                    message="No todo file found. Use 'add' action to create your first step.",
                )

            # 创建所需对象。
            export_dir = os.path.dirname(export_path)
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)

            # 持久化相关数据。
            await write_text_file(export_path, content)

            return ToolResponse(
                success=True,
                message=f"✅ Successfully exported todo.md to: {export_path}",
                extra=ToolExtra(
                    file_path=[todo.todo_file, export_path],
                    data={
                        "source_file": todo.todo_file,
                        "export_path": export_path,
                        "file_size": len(content),
                        "total_steps": len(todo.steps),
                    },
                ),
            )

        except Exception as e:
            return ToolResponse(
                success=False, message=f"Error exporting todo.md: {e!s}"
            )

    def get_todo_content(self, ctx: SessionContext, **kwargs) -> str:
        """获取与 `get_todo_content` 对应的数据或状态。"""
        id = ctx.id
        if id not in self._todo_cache:
            return "[Current todo.md is empty, fill it with your plan when applicable]"
        return self._todo_cache[id].get_content()
