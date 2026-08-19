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

from src.application import ResearchRequest, resolve_research_task
from src.logger import logger
from src.research_runner import (
    ResearchCancelledError,
    ResearchRunError,
    run_research,
)


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
            model_provider=args.model_provider,
            model_id=args.model_id,
            fallback_models=args.fallback_models,
            embedding_provider=args.embedding_provider,
            embedding_model_id=args.embedding_model_id,
        )
    except ValueError as error:
        raise SystemExit(f"研究参数无效：{error}") from error

    try:
        result = await run_research(
            research_request,
            config_path=args.config,
            cfg_options=args.cfg_options,
        )
    except ResearchCancelledError as error:
        raise SystemExit(str(error)) from error
    except ResearchRunError as error:
        raise SystemExit(f"研究运行失败：{error}") from error

    logger.info(f"| 📄 最终报告：\n{result.report}")


if __name__ == "__main__":
    asyncio.run(main())
