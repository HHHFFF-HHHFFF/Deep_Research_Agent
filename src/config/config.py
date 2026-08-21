import os
from argparse import Namespace

from dotenv import load_dotenv
from mmengine import Config as MMConfig

from src.utils import Singleton, assemble_project_path


def _normalize_model_name(provider: str, model_id: str) -> str:
    """生成统一的“提供方/模型”标识。"""
    return model_id if "/" in model_id else f"{provider}/{model_id}"


def _split_model_list(value: str) -> list[str]:
    """解析英文逗号分隔的备用模型列表。"""
    return [item.strip() for item in value.split(",") if item.strip()]


def get_environment_model_options() -> dict[str, object]:
    """读取模型环境变量，配置文件与命令行仍可继续覆盖。"""
    # Web 入口不会像命令行脚本那样预先加载 .env，因此在读取前统一加载。
    load_dotenv(verbose=False)
    scalar_mapping = {
        "MODEL_PROVIDER": "model_provider",
        "MODEL_NAME": "model_id",
        "EMBEDDING_PROVIDER": "embedding_provider",
        "EMBEDDING_MODEL": "embedding_model_id",
    }
    options: dict[str, object] = {
        config_key: value
        for env_key, config_key in scalar_mapping.items()
        if (value := os.getenv(env_key))
    }
    if fallback_models := os.getenv("MODEL_FALLBACKS"):
        options["fallback_models"] = _split_model_list(fallback_models)
    if embedding_fallbacks := os.getenv("EMBEDDING_FALLBACKS"):
        options["embedding_fallback_models"] = _split_model_list(embedding_fallbacks)
    return options


def process_models(config: MMConfig) -> MMConfig:
    """在命令行参数合并后重新计算模型名称。"""
    config.model_name = _normalize_model_name(config.model_provider, config.model_id)
    config.embedding_model_name = _normalize_model_name(
        config.embedding_provider,
        config.embedding_model_id,
    )
    return config


def process_general(config: MMConfig) -> MMConfig:
    """规范化工作目录和日志路径。"""
    workdir = str(assemble_project_path(config.workdir))
    os.makedirs(workdir, exist_ok=True)
    config.workdir = workdir

    log_path = getattr(config, "log_path", "agent.log")
    log_path = str(assemble_project_path(os.path.join(workdir, log_path)))
    config.log_path = log_path

    return config


def process_tools(config: MMConfig) -> MMConfig:
    for key in config:
        if "tool" in key:
            if "base_dir" in config[key]:
                base_dir = str(
                    assemble_project_path(
                        os.path.join(config.workdir, config[key]["base_dir"])
                    )
                )
                config[key].update({"base_dir": base_dir})
            if "model_name" in config[key]:
                config[key].update({"model_name": config.model_name})
    return config


def process_environments(config: MMConfig) -> MMConfig:
    for key in config:
        if "environment" in key:
            if "base_dir" in config[key]:
                base_dir = str(
                    assemble_project_path(
                        os.path.join(config.workdir, config[key]["base_dir"])
                    )
                )
                config[key].update({"base_dir": base_dir})
            if "embedding_model_name" in config[key]:
                config[key].update(
                    {"embedding_model_name": config.embedding_model_name}
                )
    return config


def process_memory(config: MMConfig) -> MMConfig:
    for key in config:
        if "memory" in key:
            if "base_dir" in config[key]:
                base_dir = str(
                    assemble_project_path(
                        os.path.join(config.workdir, config[key]["base_dir"])
                    )
                )
                config[key].update({"base_dir": base_dir})
            if "model_name" in config[key]:
                config[key].update({"model_name": config.model_name})
    return config


def process_agent(config: MMConfig) -> MMConfig:
    for key in config:
        if not key.endswith("_agent") or not isinstance(config[key], dict):
            continue
        if "workdir" in config[key]:
            config[key].update({"workdir": str(assemble_project_path(config.workdir))})
        if "model_name" in config[key]:
            config[key].update({"model_name": config.model_name})
    return config


class Config(MMConfig, metaclass=Singleton):
    def __init__(self):
        super().__init__()

    def initialize(self, config_path: str, args: Namespace) -> None:
        config_path = str(assemble_project_path(config_path))
        mmconfig = MMConfig.fromfile(filename=config_path)
        mmconfig.merge_from_dict(get_environment_model_options())
        if not hasattr(args, "cfg_options") or args.cfg_options is None:
            cfg_options = {}
        else:
            cfg_options = args.cfg_options
        for item in args.__dict__:
            if (
                item not in ["config", "cfg_options"]
                and args.__dict__[item] is not None
            ):
                cfg_options[item] = args.__dict__[item]

        mmconfig.merge_from_dict(cfg_options)

        mmconfig = process_models(mmconfig)
        mmconfig = process_general(mmconfig)
        mmconfig = process_tools(mmconfig)
        mmconfig = process_environments(mmconfig)
        mmconfig = process_memory(mmconfig)
        mmconfig = process_agent(mmconfig)
        print(mmconfig.pretty_text)

        self.__dict__.update(mmconfig.__dict__)

    def dump(self) -> str:
        """实现 `dump` 的业务逻辑。"""
        return super().dump()


config = Config()
config.initialize(config_path="configs/base.py", args=Namespace())
