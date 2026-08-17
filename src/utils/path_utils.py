import os
from typing import Union

def get_project_root() -> str:
    """获取与 `get_project_root` 对应的数据或状态。"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def assemble_project_path(path: str) -> str:
    """实现 `assemble_project_path` 的业务逻辑。"""
    if os.path.isabs(path):
        return os.path.abspath(path)
    else:
        return os.path.abspath(os.path.join(get_project_root(), path))
