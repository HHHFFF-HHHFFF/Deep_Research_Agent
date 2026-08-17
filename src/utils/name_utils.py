import hashlib
from typing import Optional

def get_tag_name(tag:Optional[str] = None,
                 assets_name: Optional[str] = None,
                 source: Optional[str] = None,
                 data_type: Optional[str] = None,
                 level: Optional[str] = None) -> str:
    """获取与 `get_tag_name` 对应的数据或状态。"""
    # 说明相关实现细节。

    filters = [assets_name, source, data_type, level]
    name = "_".join([str(f) for f in filters if f is not None])
    return name if tag is None else tag

def get_newspage_name(symbol: str, timestamp: str, title: str) -> str:
    """获取与 `get_newspage_name` 对应的数据或状态。"""
    return hashlib.md5(f'{symbol} {timestamp} {title}'.encode()).hexdigest()

def get_md5(text: str) -> str:
    """获取与 `get_md5` 对应的数据或状态。"""
    return hashlib.md5(text.encode()).hexdigest()
