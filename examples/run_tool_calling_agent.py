import os
import sys
import json
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
import argparse
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.model import model_manager
from src.version import version_manager
from src.prompt import prompt_manager
from src.memory import memory_manager
from src.tool import tcp
from src.skill import scp
from src.environment import ecp
from src.agent import acp
from src.session.types import SessionContext
from src.utils import generate_unique_id

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

async def main():
    args = parse_args()

    config.initialize(config_path = args.config, args = args)
    logger.initialize(config = config)
    logger.info(f"| Config: {config.pretty_text}")

    # 初始化相关状态。
    logger.info("| 🧠 Initializing model manager...")
    await model_manager.initialize()
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
    skill_names = getattr(config, 'skill_names', None)
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
    logger.info(f"| ✅ Version manager initialized: {json.dumps(await version_manager.list(), indent=4)}")

    # 检索所需信息。
    task = "调研大模型 Agent 的提示注入风险与主流防护方法，并生成带来源的中文报告。"
    files = []

    logger.info(f"| 📋 Task: {task}")
    logger.info(f"| 📂 Files: {files}")

    # 说明相关实现细节。
    ctx = SessionContext()

    input = {
        "name": "tool_calling",
        "input": {
            "task": task,
            "files": files
        },
        "ctx": ctx
    }
    await acp(**input)

if __name__ == "__main__":
    asyncio.run(main())
