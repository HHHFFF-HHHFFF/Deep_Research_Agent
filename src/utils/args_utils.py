"""提供args utils相关实现。"""

import json
import re
from typing import Dict, Any

import dirtyjson

# 说明相关实现细节。


def parse_tool_args(args_str: str) -> Dict[str, Any]:
    """解析与 `parse_tool_args` 对应的数据或状态。"""
    if not args_str:
        return {}

    # 说明相关实现细节。
    try:
        return dirtyjson.loads(args_str)
    except (dirtyjson.Error, ValueError, TypeError) as e:
        pass

    # 说明相关实现细节。
    try:
        return json.loads(args_str)
    except json.JSONDecodeError as e:
        pass

    # 说明相关实现细节。
    # 说明相关实现细节。
    try:
        # 说明相关实现细节。
        # 说明相关实现细节。
        fixed_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', args_str)
        return json.loads(fixed_str)
    except (json.JSONDecodeError, re.error) as e:
        pass

    # 说明相关实现细节。
    try:
        result = {}
        # 说明相关实现细节。
        pattern = r'"(\w+)"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(\d+(?:\.\d+)?))'
        matches = re.findall(pattern, args_str)
        for key, str_val, num_val in matches:
            if str_val:
                # 说明相关实现细节。
                result[key] = str_val.encode().decode('unicode_escape', errors='replace')
            elif num_val:
                result[key] = int(num_val) if '.' not in num_val else float(num_val)
        if result:
            return result
    except Exception as e:
        pass

    # 处理异常情况。
    return {}
