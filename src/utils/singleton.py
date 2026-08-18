"""提供singleton相关实现。"""

import abc
from typing import ClassVar


class Singleton(abc.ABCMeta, type):
    """定义 `Singleton`，封装相关数据与行为。"""

    _instances: ClassVar[dict[type, object]] = {}

    def __call__(cls, *args, **kwargs):
        """执行组件调用并返回结果。"""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractSingleton(abc.ABC, metaclass=Singleton):
    """定义 `AbstractSingleton`，封装相关数据与行为。"""
