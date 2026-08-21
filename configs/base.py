# 模型配置可由配置加载器读取 `.env` 或命令行后覆盖。
tag = "base"
workdir = f"workdir/{tag}"
log_path = "base.log"
use_local_proxy = False

model_provider = "qwen"
model_id = "qwen3-max"
model_name = model_id if "/" in model_id else f"{model_provider}/{model_id}"
fallback_models: list[str] = []

embedding_provider = "qwen"
embedding_model_id = "text-embedding-v4"
embedding_model_name = (
    embedding_model_id
    if "/" in embedding_model_id
    else f"{embedding_provider}/{embedding_model_id}"
)
embedding_fallback_models: list[str] = []

# 配置相关参数。
memory_config = {
    "type": "general_memory_system",
    "model_name": model_name,
    "max_summaries": 20,
    "max_insights": 100,
}

# 模型最大输出长度。
max_tokens = 16384

# 本地界面尺寸，当前仅保留兼容配置。
window_size = (1024, 768)
