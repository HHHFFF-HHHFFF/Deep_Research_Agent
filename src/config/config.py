import os
from mmengine import Config as MMConfig
from argparse import Namespace
from typing import Union

from src.utils import assemble_project_path, Singleton

def process_general(config: MMConfig) -> MMConfig:
    """实现 `process_general` 的业务逻辑。"""
    workdir = str(assemble_project_path(config.workdir))
    os.makedirs(workdir, exist_ok=True)
    config.workdir = workdir

    log_path = getattr(config, 'log_path', 'agent.log')
    log_path = str(assemble_project_path(os.path.join(workdir, log_path)))
    config.log_path = log_path

    return config

def process_tools(config: MMConfig) -> MMConfig:
    for key in config:
        if "tool" in key:
            if "base_dir" in config[key]:
                # 配置相关参数。
                # 处理工具调用。
                base_dir = str(assemble_project_path(os.path.join(config.workdir, config[key]["base_dir"])))
                config[key].update(dict(
                    base_dir = base_dir
                ))
    return config

def process_environments(config: MMConfig) -> MMConfig:
    for key in config:
        if "environment" in key:
            if "base_dir" in config[key]:
                base_dir = str(assemble_project_path(os.path.join(config.workdir, config[key]["base_dir"])))
                config[key].update(dict(
                    base_dir = base_dir
                ))
    return config

def process_memory(config: MMConfig)->MMConfig:
    for key in config:
        if "memory" in key:
            if "base_dir" in config[key]:
                base_dir = str(assemble_project_path(os.path.join(config.workdir, config[key]["base_dir"])))
                config[key].update(dict(
                    base_dir = base_dir
                ))
            if "model_name" in config[key]:
                model_name = config.model_name
                config[key].update(
                    dict(
                        model_name = model_name
                    )
                )
    return config

def process_agent(config: MMConfig) -> MMConfig:
    if "agent" in config:
        if "workdir" in config.agent:
            # 配置相关参数。
            config.agent.update(dict(
                workdir = str(assemble_project_path(config.workdir))
            ))
        if "model_name" in config.agent:
            config.agent.update(dict(
                model_name = config.model_name
            ))
    return config

class Config(MMConfig, metaclass=Singleton):
    def __init__(self):
        super(Config, self).__init__()

    def initialize(self, config_path: Union[str], args: Namespace) -> None:
        # 配置相关参数。
        config_path = str(assemble_project_path(config_path))
        mmconfig = MMConfig.fromfile(filename=config_path)
        if 'cfg_options' not in args or args.cfg_options is None:
            cfg_options = dict()
        else:
            cfg_options = args.cfg_options
        for item in args.__dict__:
            if item not in ['config', 'cfg_options'] and args.__dict__[item] is not None:
                cfg_options[item] = args.__dict__[item]

        mmconfig.merge_from_dict(cfg_options)

        # 配置相关参数。
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
