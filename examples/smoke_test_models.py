"""使用真实 API 验证聊天模型与向量模型连接。"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.message import HumanMessage
from src.model.runtime_manager import ModelManager
from src.model.settings import ModelRuntimeSettings


async def main() -> int:
    """执行不输出密钥和原始响应的最小真实调用。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(verbose=False)
    settings = ModelRuntimeSettings.from_env()
    manager = ModelManager()
    await manager.initialize(settings)

    print(f"聊天模型：{settings.primary_model}")
    chat_result = await manager.achat(
        [HumanMessage(content="请只回复：连接成功")]
    )
    if chat_result.success:
        print(f"聊天测试：成功，响应={chat_result.message[:100]}")
    else:
        print(f"聊天测试：失败，原因={chat_result.message[:500]}")

    print(f"向量模型：{settings.embedding_model}")
    embedding_result = await manager.aembedding(
        [HumanMessage(content="深度研究智能体向量连接测试")]
    )
    embedding_succeeded = False
    if embedding_result.success:
        data = embedding_result.extra.data if embedding_result.extra else {}
        embeddings = data.get("embeddings", []) if data else []
        dimension = len(embeddings[0]) if embeddings else 0
        embedding_succeeded = bool(embeddings and dimension)
        status = "成功" if embedding_succeeded else "失败，响应中没有向量"
        print(f"向量测试：{status}，数量={len(embeddings)}，维度={dimension}")
    else:
        print(f"向量测试：失败，原因={embedding_result.message[:500]}")

    return 0 if chat_result.success and embedding_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
