"""通用工具的按需导出，避免导入时加载所有可选依赖。"""

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "open_binary_file": (".async_file", "open_binary_file"),
    "read_json_file": (".async_file", "read_json_file"),
    "read_lines_file": (".async_file", "read_lines_file"),
    "read_text_file": (".async_file", "read_text_file"),
    "write_json_file": (".async_file", "write_json_file"),
    "write_lines_file": (".async_file", "write_lines_file"),
    "write_pickle_file": (".async_file", "write_pickle_file"),
    "write_text_file": (".async_file", "write_text_file"),
    "get_project_root": (".path_utils", "get_project_root"),
    "assemble_project_path": (".path_utils", "assemble_project_path"),
    "Singleton": (".singleton", "Singleton"),
    "_is_package_available": (".utils", "_is_package_available"),
    "encode_file_base64": (".utils", "encode_file_base64"),
    "decode_file_base64": (".utils", "decode_file_base64"),
    "make_file_url": (".utils", "make_file_url"),
    "parse_json_blob": (".utils", "parse_json_blob"),
    "gather_with_concurrency": (".utils", "gather_with_concurrency"),
    "get_token_count": (".token_utils", "get_token_count"),
    "extract_boxed_content": (".string_utils", "extract_boxed_content"),
    "dedent": (".string_utils", "dedent"),
    "generate_unique_id": (".string_utils", "generate_unique_id"),
    "get_tag_name": (".name_utils", "get_tag_name"),
    "get_newspage_name": (".name_utils", "get_newspage_name"),
    "get_md5": (".name_utils", "get_md5"),
    "fetch_url": (".url_utils", "fetch_url"),
    "get_file_info": (".file_utils", "get_file_info"),
    "file_lock": (".file_utils", "file_lock"),
    "get_env": (".env_utils", "get_env"),
    "parse_tool_args": (".args_utils", "parse_tool_args"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """首次访问某个工具时再加载其所在模块。"""
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
