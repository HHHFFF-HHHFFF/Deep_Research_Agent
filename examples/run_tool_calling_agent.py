import json
import os
import sys

from dotenv import load_dotenv

load_dotenv(verbose=True)

import argparse
import asyncio
from pathlib import Path

from mmengine import DictAction

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.agent import acp
from src.application import ResearchRequest, resolve_research_task
from src.config import config
from src.environment import ecp
from src.logger import logger
from src.memory import memory_manager
from src.model import model_manager
from src.prompt import prompt_manager
from src.session.types import SessionContext
from src.skill import scp
from src.tool import tcp
from src.version import version_manager


def parse_args():
    parser = argparse.ArgumentParser(description="运行深度研究智能体")
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs", "tool_calling_agent.py"),
        help="配置文件路径",
    )
    parser.add_argument(
        "--task",
        help="需要研究的问题或方向；未提供时将在终端中交互输入",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        help="需要用于本地 RAG 的文档路径，可重复传入",
    )
    parser.add_argument(
        "--provider",
        dest="model_provider",
        help="聊天模型提供方，例如 qwen 或 deepseek",
    )
    parser.add_argument("--model", dest="model_id", help="聊天模型标识，例如 qwen-plus")
    parser.add_argument(
        "--fallback-model",
        dest="fallback_models",
        action="append",
        help="备用模型，使用“提供方/模型”格式，可重复传入",
    )
    parser.add_argument("--embedding-provider", help="向量模型提供方")
    parser.add_argument(
        "--embedding-model", dest="embedding_model_id", help="向量模型标识"
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="使用 key=value 形式覆盖配置项",
    )
    args = parser.parse_args()
    return args


async def main():
    args = parse_args()

    try:
        research_request = ResearchRequest(
            task=resolve_research_task(args.task),
            files=args.files or [],
        )
    except ValueError as error:
        raise SystemExit(f"研究主题无效：{error}") from error

    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
    logger.info(f"| Config: {config.pretty_text}")

    # 初始化相关状态。
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize(
        primary_model=config.model_name,
        fallback_models=config.fallback_models,
        embedding_model=config.embedding_model_name,
        embedding_fallback_models=config.embedding_fallback_models,
    )
    logger.info(f"| ✅ Model manager initialized: {await model_manager.list()}")

    # 初始化相关状态。
    logger.info("| 📁 Initializing prompt manager...")
    await prompt_manager.initialize()
    logger.info(f"| ✅ Prompt manager initialized: {await prompt_manager.list()}")

    # 初始化相关状态。
    logger.info("| 📁 Initializing memory manager...")
    await memory_manager.initialize(memory_names=config.memory_names)
    logger.info(f"| ✅ Memory manager initialized: {await memory_manager.list()}")

    # 初始化相关状态。
    logger.info("| 🛠️ Initializing tools...")
    await tcp.initialize(tool_names=config.tool_names)
    logger.info(f"| ✅ Tools initialized: {await tcp.list()}")

    # 初始化相关状态。
    logger.info("| 🎯 Initializing skills...")
    skill_names = getattr(config, "skill_names", None)
    await scp.initialize(skill_names=skill_names)
    logger.info(f"| ✅ Skills initialized: {await scp.list()}")

    # 初始化相关状态。
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")

    # 初始化相关状态。
    logger.info("| 🤖 Initializing agents...")
    await acp.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ Agents initialized: {await acp.list()}")

    # 初始化相关状态。
    logger.info("| 📁 Initializing version manager...")
    await version_manager.initialize()
    logger.info(
        f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}"
    )

    logger.info(f"| 📋 Task: {research_request.task}")
    logger.info(f"| 📂 Files: {research_request.files}")

    # 说明相关实现细节。
    ctx = SessionContext()

    agent_input = {
        "name": "tool_calling",
        "input": {
            "task": research_request.task,
            "files": research_request.files,
        },
        "ctx": ctx,
    }
    await acp(**agent_input)


if __name__ == "__main__":
    asyncio.run(main())
