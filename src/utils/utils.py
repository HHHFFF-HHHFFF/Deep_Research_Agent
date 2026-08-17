#!/usr/bin/env python
# 说明相关实现细节。

# 说明相关实现细节。
#
# 处理版本与历史记录。
# 处理文件与路径。
# 说明相关实现细节。
#
# 说明相关实现细节。
#
# 说明相关实现细节。
# 说明相关实现细节。
# 说明相关实现细节。
# 说明相关实现细节。
# 说明相关实现细节。
import ast
import asyncio
import base64
import importlib.metadata
import importlib.util
import inspect
import mimetypes
import json
import json5
import keyword
import os
import re
import types
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Tuple, Optional, Union, Iterable, Awaitable, List, TypeVar

T = TypeVar("T")

@lru_cache
def _is_package_available(package_name: str) -> bool:
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def escape_code_brackets(text: str) -> str:
    """实现 `escape_code_brackets` 的业务逻辑。"""

    def replace_bracketed_content(match):
        content = match.group(1)
        cleaned = re.sub(
            r"bold|red|green|blue|yellow|magenta|cyan|white|black|italic|dim|\s|#[0-9a-fA-F]{6}", "", content
        )
        return f"\\[{content}\\]" if cleaned.strip() else f"[{content}]"

    return re.sub(r"\[([^\]]*)\]", replace_bracketed_content, text)


def make_json_serializable(obj: Any) -> Any:
    """实现 `make_json_serializable` 的业务逻辑。"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # 转换并规范化数据。
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (obj.startswith("[") and obj.endswith("]")):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        # 转换并规范化数据。
        return {"_type": obj.__class__.__name__, **{k: make_json_serializable(v) for k, v in obj.__dict__.items()}}
    else:
        # 转换并规范化数据。
        return str(obj)


def parse_json_blob(json_blob: str) -> Tuple[Dict[str, str], str]:
    """解析与 `parse_json_blob` 对应的数据或状态。"""
    try:
        if "Calling tools:" in json_blob:
            json_blob = json_blob.split("Calling tools:")[-1]

        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_data = json_blob[first_accolade_index: last_accolade_index + 1]

        json_data = json5.loads(json_data, strict=False)
        json_data = json_data['function']

        return json_data, json_blob[:first_accolade_index]
    except IndexError:
        raise ValueError("The model output does not contain any JSON blob.")
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place - 4 : place + 5]}'."
        )


def extract_code_from_text(text: str) -> Optional[str]:
    """提取与 `extract_code_from_text` 对应的数据或状态。"""
    pattern = r"<code>(.*?)</code>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None


def parse_code_blobs(text: str) -> str:
    """解析与 `parse_code_blobs` 对应的数据或状态。"""
    matches = extract_code_from_text(text)
    if matches:
        return matches
    # 组装并返回结果。
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass

    if "final" in text and "answer" in text:
        raise ValueError(
            dedent(
                f"""
                Your code snippet is invalid, because the regex pattern <code>(.*?)</code> was not found in it.
                Here is your code snippet:
                {text}
                It seems like you're trying to return the final answer, you can do it as follows:
                <code>
                final_answer("YOUR FINAL ANSWER HERE")
                </code>
                """
            ).strip()
        )
    raise ValueError(
        dedent(
            f"""
            Your code snippet is invalid, because the regex pattern <code>(.*?)</code> was not found in it.
            Here is your code snippet:
            {text}
            Make sure to include code with the correct pattern, for instance:
            Thoughts: Your thoughts
            <code>
            # Your python code here
            </code>
            """
        ).strip()
    )


async def gather_with_concurrency(
    coros: Iterable[Awaitable[T]],
    max_concurrency: int = 10,
    return_exceptions: bool = False,
) -> List[Union[T, BaseException]]:
    """实现 `gather_with_concurrency` 的业务逻辑。"""
    sem = asyncio.Semaphore(max_concurrency)

    async def _runner(coro: Awaitable[T]) -> Union[T, BaseException]:
        async with sem:
            if return_exceptions:
                try:
                    return await coro
                except BaseException as e:  # noqa: BLE001
                    return e
            else:
                return await coro

    return await asyncio.gather(
        *(_runner(c) for c in coros),
        return_exceptions=False,  # 说明相关实现细节。
    )


class ImportFinder(ast.NodeVisitor):
    def __init__(self):
        self.packages = set()

    def visit_Import(self, node):
        for alias in node.names:
            # 说明相关实现细节。
            base_package = alias.name.split(".")[0]
            self.packages.add(base_package)

    def visit_ImportFrom(self, node):
        if node.module:  # 说明相关实现细节。
            # 说明相关实现细节。
            base_package = node.module.split(".")[0]
            self.packages.add(base_package)


def get_method_source(method):
    """获取与 `get_method_source` 对应的数据或状态。"""
    if isinstance(method, types.MethodType):
        method = method.__func__
    return get_source(method)


def is_same_method(method1, method2):
    """实现 `is_same_method` 的业务逻辑。"""
    try:
        source1 = get_method_source(method1)
        source2 = get_method_source(method2)

        # 移除相关数据或组件。
        source1 = "\n".join(line for line in source1.split("\n") if not line.strip().startswith("@"))
        source2 = "\n".join(line for line in source2.split("\n") if not line.strip().startswith("@"))

        return source1 == source2
    except (TypeError, OSError):
        return False


def is_same_item(item1, item2):
    """实现 `is_same_item` 的业务逻辑。"""
    if callable(item1) and callable(item2):
        return is_same_method(item1, item2)
    else:
        return item1 == item2


def instance_to_source(instance, base_cls=None):
    """实现 `instance_to_source` 的业务逻辑。"""
    cls = instance.__class__
    class_name = cls.__name__

    # 创建所需对象。
    class_lines = []
    if base_cls:
        class_lines.append(f"class {class_name}({base_cls.__name__}):")
    else:
        class_lines.append(f"class {class_name}:")

    # 说明相关实现细节。
    if cls.__doc__ and (not base_cls or cls.__doc__ != base_cls.__doc__):
        class_lines.append(f'    """{cls.__doc__}"""')

    # 说明相关实现细节。
    class_attrs = {
        name: value
        for name, value in cls.__dict__.items()
        if not name.startswith("__")
        and not callable(value)
        and not (base_cls and hasattr(base_cls, name) and getattr(base_cls, name) == value)
    }

    for name, value in class_attrs.items():
        if isinstance(value, str):
            # 说明相关实现细节。
            if "\n" in value:
                escaped_value = value.replace('"""', r"\"\"\"")  # 说明相关实现细节。
                class_lines.append(f'    {name} = """{escaped_value}"""')
            else:
                class_lines.append(f"    {name} = {json.dumps(value)}")
        else:
            class_lines.append(f"    {name} = {repr(value)}")

    if class_attrs:
        class_lines.append("")

    # 说明相关实现细节。
    methods = {
        name: func.__wrapped__ if hasattr(func, "__wrapped__") else func
        for name, func in cls.__dict__.items()
        if callable(func)
        and (
            not base_cls
            or not hasattr(base_cls, name)
            or (
                isinstance(func, (staticmethod, classmethod))
                or (getattr(base_cls, name).__code__.co_code != func.__code__.co_code)
            )
        )
    }

    for name, method in methods.items():
        method_source = get_source(method)
        # 说明相关实现细节。
        method_lines = method_source.split("\n")
        first_line = method_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        method_lines = [line[indent:] for line in method_lines]
        method_source = "\n".join(["    " + line if line.strip() else line for line in method_lines])
        class_lines.append(method_source)
        class_lines.append("")

    # 说明相关实现细节。
    import_finder = ImportFinder()
    import_finder.visit(ast.parse("\n".join(class_lines)))
    required_imports = import_finder.packages

    # 创建所需对象。
    final_lines = []

    # 说明相关实现细节。
    if base_cls:
        final_lines.append(f"from {base_cls.__module__} import {base_cls.__name__}")

    # 说明相关实现细节。
    for package in required_imports:
        final_lines.append(f"import {package}")

    if final_lines:  # 说明相关实现细节。
        final_lines.append("")

    # 说明相关实现细节。
    final_lines.extend(class_lines)

    return "\n".join(final_lines)


def get_source(obj) -> str:
    """获取与 `get_source` 对应的数据或状态。"""
    if not (isinstance(obj, type) or callable(obj)):
        raise TypeError(f"Expected class or callable, got {type(obj)}")

    inspect_error = None
    try:
        # 创建所需对象。
        source = getattr(obj, "__source__", None) or inspect.getsource(obj)
        return dedent(source).strip()
    except OSError as e:
        # 处理异常情况。
        inspect_error = e
    try:
        import IPython

        shell = IPython.get_ipython()
        if not shell:
            raise ImportError("No active IPython shell found")
        all_cells = "\n".join(shell.user_ns.get("In", [])).strip()
        if not all_cells:
            raise ValueError("No code cells found in IPython session")

        tree = ast.parse(all_cells)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == obj.__name__:
                return dedent("\n".join(all_cells.split("\n")[node.lineno - 1 : node.end_lineno])).strip()
        raise ValueError(f"Could not find source code for {obj.__name__} in IPython history")
    except ImportError:
        # 处理异常情况。
        raise inspect_error
    except ValueError as e:
        # 处理异常情况。
        raise e from inspect_error


def encode_file_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

def decode_file_base64(data_base64: str) -> bytes:
    return base64.b64decode(data_base64)

def make_file_url(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    return f"data:{mime_type.lower()};base64,{encode_file_base64(file_path)}"


def make_init_file(folder: Union[str, Path]):
    os.makedirs(folder, exist_ok=True)
    # 创建所需对象。
    with open(os.path.join(folder, "__init__.py"), "w"):
        pass


def is_valid_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name) if isinstance(name, str) else False
