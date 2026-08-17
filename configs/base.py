# 配置相关参数。
tag = "base"
workdir = f"workdir/{tag}"
log_path = "base.log"
use_local_proxy = False
model_name = "openrouter/gemini-3-flash-preview"

# 配置相关参数。
memory_config = dict(
    type = "general_memory_system",
    model_name = "gpt-4.1",
    max_summaries = 20,
    max_insights = 100
)

# 配置相关参数。
max_tokens = 16384

# 配置相关参数。
window_size = (1024, 768)

# 配置相关参数。
alpaca_service = dict(
    base_dir=workdir,
    accounts=None,
    live=False,
    auto_start_data_stream=True,
    symbol=["BTC/USD"],
    data_type=["bars"],
)

# 配置相关参数。
binance_service = dict(
    base_dir=workdir,
    accounts=None,
    live=False,
    auto_start_data_stream=True,
    symbol=["BTCUSDT"],
    data_type=["klines"],
)
