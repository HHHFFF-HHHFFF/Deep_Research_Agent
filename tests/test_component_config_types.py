import json

from src.environment.types import ActionResult, EnvironmentConfig
from src.memory.types import MemoryConfig
from src.prompt.types import PromptConfig
from src.tool.types import ToolConfig


def test_tool_config_from_dict_does_not_restore_runtime_objects() -> None:
    config = ToolConfig.from_dict(
        {
            "name": "search",
            "description": "搜索工具",
            "cls": "dangerous.module.Tool",
            "instance": {"unexpected": True},
        }
    )

    assert config.name == "search"
    assert config.cls is None
    assert config.instance is None


def test_memory_config_from_dict_clears_runtime_objects() -> None:
    memory_config = MemoryConfig.from_dict(
        {
            "name": "memory",
            "description": "研究记忆",
            "cls": "dangerous.module.Memory",
            "instance": {"unexpected": True},
        }
    )

    assert memory_config.cls is None
    assert memory_config.instance is None


def test_environment_config_restores_static_actions_only() -> None:
    config = EnvironmentConfig.from_dict(
        {
            "name": "filesystem",
            "description": "文件环境",
            "rules": "仅允许访问工作目录",
            "cls": "dangerous.module.Environment",
            "actions": {
                "read": {
                    "env_name": "filesystem",
                    "name": "read",
                    "description": "读取文件",
                    "function": "dangerous.module.read",
                }
            },
        }
    )

    assert config.cls is None
    assert config.actions["read"].function is None


def test_prompt_config_from_dict_keeps_pydantic_validation() -> None:
    config = PromptConfig.from_dict(
        {
            "name": "research",
            "type": "system_prompt",
            "description": "研究提示词",
            "template": "请研究：{task}",
            "cls": "dangerous.module.Prompt",
        }
    )

    assert config.name == "research"
    assert config.cls is None


def test_action_result_uses_pydantic_json_serialization() -> None:
    result = ActionResult(success=True, message="完成", extra={"count": 1})

    assert json.loads(result.model_dump_json()) == {
        "success": True,
        "message": "完成",
        "extra": {"count": 1},
    }
