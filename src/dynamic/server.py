"""提供服务入口相关实现。"""

import ast
import inspect
import json
import re
from collections.abc import Callable
from copy import deepcopy
from typing import (
    Any,
    TypeVar,
    Union,
    get_type_hints,
)

import inflection
from pydantic import BaseModel, ConfigDict, Field, create_model

T = TypeVar("T")

# 说明相关实现细节。
PYTHON_TYPE_FIELD = "x-python-type"
JSON_TO_PYTHON_TYPE = {
    "integer": "int",
    "number": "float",
    "string": "str",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
}


class DynamicCodeExecutionDisabledError(RuntimeError):
    """表示项目安全策略禁止执行运行时生成的 Python 代码。"""


class DynamicModuleManager:
    """定义 `DynamicModuleManager`，封装相关数据与行为。"""

    def __init__(self):
        """初始化实例。"""
        self._module_counter = 0
        self._loaded_modules: dict[str, Any] = {}  # 说明相关实现细节。
        # 说明相关实现细节。
        self._symbol_registry: dict[str, Any] = {}
        # 组装并返回结果。
        self._context_providers: dict[str, Callable[[], dict[str, Any]]] = {}

    def default_parameters_schema(self) -> dict[str, Any]:
        """实现 `default_parameters_schema` 的业务逻辑。"""
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def parse_docstring_descriptions(self, docstring: str) -> dict[str, str]:
        """解析与 `parse_docstring_descriptions` 对应的数据或状态。"""
        if not docstring:
            return {}

        descriptions: dict[str, str] = {}
        lines = inspect.cleandoc(docstring).splitlines()
        in_args = False

        for line in lines:
            stripped = line.strip()
            if not in_args:
                if stripped.lower().startswith("args"):
                    in_args = True
                continue

            if stripped.lower().startswith(
                ("returns:", "yields:", "raises:", "examples:")
            ):
                break

            match = re.match(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", stripped)
            if match:
                param_name = match.group(1)
                description = match.group(2).strip()
                descriptions[param_name] = description

        return descriptions

    def annotation_to_types(self, annotation: Any) -> tuple[str, str]:
        """实现 `annotation_to_types` 的业务逻辑。"""
        if annotation is inspect._empty or annotation is None:
            return "string", "Any"

        basic_map = {
            str: ("string", "str"),
            int: ("integer", "int"),
            float: ("number", "float"),
            bool: ("boolean", "bool"),
            dict: ("object", "dict"),
            list: ("array", "list"),
        }
        if annotation in basic_map:
            return basic_map[annotation]

        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())

        if origin is Union and len(args) == 2 and type(None) in args:
            inner_type = args[0] if args[1] is type(None) else args[1]
            json_type, python_type = self.annotation_to_types(inner_type)
            return json_type, f"Optional[{python_type}]"

        if origin is list or (
            hasattr(annotation, "__origin__") and "List" in str(annotation)
        ):
            return "array", "list"

        if origin is dict or (
            hasattr(annotation, "__origin__") and "Dict" in str(annotation)
        ):
            return "object", "dict"

        type_str = str(annotation).replace("typing.", "")
        return "string", type_str

    def parse_type_string(self, type_str: str) -> type:
        """解析与 `parse_type_string` 对应的数据或状态。"""
        # 移除相关数据或组件。
        type_str = type_str.replace("typing.", "").strip()

        # 说明相关实现细节。
        mapping = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": float,
            "bool": bool,
            "boolean": bool,
            "dict": dict,
            "object": dict,
            "list": list,
            "array": list,
            "Any": Any,
        }
        if type_str in mapping:
            return mapping[type_str]

        # 说明相关实现细节。
        if type_str.startswith("Optional[") and type_str.endswith("]"):
            inner = type_str[9:-1].strip()
            return self.parse_type_string(inner) | None

        # 说明相关实现细节。
        if type_str.startswith("List[") and type_str.endswith("]"):
            inner = type_str[5:-1].strip()
            return list[self.parse_type_string(inner)]  # type: ignore[index]

        # 转换并规范化数据。
        if type_str.startswith("Dict[") and type_str.endswith("]"):
            inner = type_str[5:-1].strip()
            # 转换并规范化数据。
            if "," in inner:
                parts = inner.split(",", 1)
                if len(parts) == 2:
                    key_type = self.parse_type_string(parts[0].strip())
                    value_type = self.parse_type_string(parts[1].strip())
                    return dict[key_type, value_type]  # type: ignore[index]
            # 处理异常情况。
            return dict

        # 转换并规范化数据。
        return Any

    def json_type_to_python_type(self, json_type: str) -> type:
        """实现 `json_type_to_python_type` 的业务逻辑。"""
        mapping = {
            "integer": int,
            "number": float,
            "string": str,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        return mapping.get(json_type, str)

    def remove_python_type_field(self, schema: dict[str, Any]) -> dict[str, Any]:
        """移除与 `remove_python_type_field` 对应的数据或状态。"""
        cleaned = deepcopy(schema)
        cleaned.pop(PYTHON_TYPE_FIELD, None)
        if "properties" in cleaned:
            for prop_info in cleaned["properties"].values():
                if isinstance(prop_info, dict):
                    prop_info.pop(PYTHON_TYPE_FIELD, None)
        return cleaned

    def build_args_schema(self, name: str, schema: dict[str, Any]) -> type[BaseModel]:
        """构建与 `build_args_schema` 对应的数据或状态。"""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        model_name = inflection.camelize(name) + "Input"

        if not properties:
            return create_model(
                model_name,
                __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow"),
            )

        field_definitions: dict[str, Any] = {}
        for param_name, param_info in properties.items():
            python_type_str = param_info.get(PYTHON_TYPE_FIELD)
            if python_type_str:
                python_type = self.parse_type_string(python_type_str)
            else:
                json_type = param_info.get("type", "string")
                python_type = self.json_type_to_python_type(json_type)

            is_required = param_name in required
            if "default" in param_info:
                default_value = param_info["default"]
            elif is_required:
                default_value = ...  # 说明相关实现细节。
            else:
                default_value = None

            description = param_info.get("description", "")
            if is_required and default_value is ...:
                field_definitions[param_name] = (
                    python_type,
                    Field(description=description),
                )
            else:
                field_definitions[param_name] = (
                    python_type | None if not is_required else python_type,
                    Field(default=default_value, description=description),
                )

        return create_model(
            model_name,
            __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow"),
            **field_definitions,
        )

    def build_function_calling(
        self, name: str, description: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        """构建与 `build_function_calling` 对应的数据或状态。"""
        # 移除相关数据或组件。
        cleaned_schema = self.remove_python_type_field(schema)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": cleaned_schema,
            },
        }

    def build_text_representation(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        entity_type: str = "Tool",
    ) -> str:
        """构建与 `build_text_representation` 对应的数据或状态。"""
        text = f"{entity_type}: {name}\nDescription: {description}\n"
        return text

    def _generate_module_name(self, prefix: str = "dynamic_module") -> str:
        """实现 `_generate_module_name` 的业务逻辑。"""
        self._module_counter += 1
        # 说明相关实现细节。
        # 处理文件与路径。
        return f"_{prefix}_{self._module_counter}"

    def is_dynamic_class(self, cls: type) -> bool:
        """实现 `is_dynamic_class` 的业务逻辑。"""
        if not hasattr(cls, "__module__"):
            return True
        module_name = cls.__module__
        # 校验输入与当前状态。
        return (
            module_name in ("__main__", "<string>", "<exec>")
            or module_name.startswith("_dynamic_")
            or "<" in module_name
        )

    def get_source_code(self, object: type["T"] | Callable) -> str | None:
        """获取与 `get_source_code` 对应的数据或状态。"""
        try:
            return inspect.getsource(object)
        except (OSError, TypeError):
            # 说明相关实现细节。
            return None

    def get_full_module_source(self, cls: type) -> str:
        """获取与 `get_full_module_source` 对应的数据或状态。"""
        try:
            # 说明相关实现细节。
            module = inspect.getmodule(cls)
            if module is None:
                # 执行回退或重试逻辑。
                return inspect.getsource(cls)

            # 处理文件与路径。
            file_path = inspect.getfile(module)

            # 加载所需数据。
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, TypeError, AttributeError):
            # 加载所需数据。
            try:
                return inspect.getsource(cls)
            except Exception:
                return ""

    def extract_class_name_from_code(self, code: str) -> str | None:
        """提取与 `extract_class_name_from_code` 对应的数据或状态。"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    return node.name
        except (SyntaxError, TypeError):
            return None
        return None

    def register_symbol(self, name: str, obj: Any) -> None:
        """注册与 `register_symbol` 对应的数据或状态。"""
        self._symbol_registry[name] = obj

    def register_context_provider(
        self, context_name: str, provider: Callable[[], dict[str, Any]]
    ) -> None:
        """注册与 `register_context_provider` 对应的数据或状态。"""
        self._context_providers[context_name] = provider

    def _extract_used_symbols(self, code: str) -> set[str]:
        """实现 `_extract_used_symbols` 的业务逻辑。"""
        used_symbols = set()

        try:
            tree = ast.parse(code)

            class SymbolCollector(ast.NodeVisitor):
                def __init__(self):
                    self.imports = set()
                    self.names = set()
                    self.in_def = False  # 说明相关实现细节。

                def visit_Import(self, node):
                    for alias in node.names:
                        self.imports.add(alias.asname or alias.name)

                def visit_ImportFrom(self, node):
                    for alias in node.names:
                        self.imports.add(alias.asname or alias.name)

                def visit_FunctionDef(self, node):
                    self.imports.add(node.name)  # 说明相关实现细节。
                    self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node):
                    self.imports.add(node.name)  # 说明相关实现细节。
                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    self.imports.add(node.name)  # 说明相关实现细节。
                    self.generic_visit(node)

                def visit_Name(self, node):
                    # 加载所需数据。
                    if isinstance(node.ctx, ast.Load):
                        self.names.add(node.id)

            collector = SymbolCollector()
            collector.visit(tree)

            # 组装并返回结果。
            # 说明相关实现细节。
            excluded = collector.imports | {
                "self",
                "cls",
                "super",
                "__name__",
                "__main__",
                "__file__",
                "__doc__",
                "True",
                "False",
                "None",
                "Exception",
                "BaseException",
                "object",
                "type",
                "str",
                "int",
                "float",
                "bool",
                "list",
                "dict",
                "tuple",
                "set",
            }
            used_symbols = collector.names - excluded

        except (SyntaxError, TypeError):
            return used_symbols

        return used_symbols

    def _auto_inject_imports(
        self, code: str, context: str | None = None
    ) -> dict[str, Any]:
        """实现 `_auto_inject_imports` 的业务逻辑。"""
        imports = {}

        # 说明相关实现细节。
        if context and context in self._context_providers:
            context_imports = self._context_providers[context]()
            imports.update(context_imports)

        # 说明相关实现细节。
        used_symbols = self._extract_used_symbols(code)

        # 注册相关组件。
        for symbol_name in used_symbols:
            if symbol_name in self._symbol_registry:
                imports[symbol_name] = self._symbol_registry[symbol_name]

        return imports

    def load_code(
        self,
        code: str,
        module_name: str | None = None,
        context: str | None = None,
        inject_imports: dict[str, Any] | None = None,
    ) -> str:
        """拒绝执行运行时生成的代码，避免绕过项目安全边界。"""
        raise DynamicCodeExecutionDisabledError(
            "当前项目禁止动态执行 Python 代码，请通过受审查的静态模块扩展能力"
        )

    def load_class(
        self,
        code: str,
        class_name: str | None = None,
        base_class: type[T] | None = None,
        module_name: str | None = None,
        context: str | None = None,
        inject_imports: dict[str, Any] | None = None,
    ) -> type[T]:
        """加载与 `load_class` 对应的数据或状态。"""
        # 说明相关实现细节。
        if context is None and base_class is not None:
            # 说明相关实现细节。
            base_name = base_class.__name__.lower()
            if "tool" in base_name:
                context = "tool"
            elif "agent" in base_name:
                context = "agent"
            elif "prompt" in base_name:
                context = "prompt"

        # 加载所需数据。
        if module_name is None:
            module_name = self.load_code(
                code, context=context, inject_imports=inject_imports
            )
        else:
            if module_name not in self._loaded_modules:
                self.load_code(
                    code, module_name, context=context, inject_imports=inject_imports
                )

        # 说明相关实现细节。
        module = self._loaded_modules[module_name]

        # 说明相关实现细节。
        if class_name is None:
            if base_class is not None:
                # 说明相关实现细节。
                candidate_classes = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, base_class)
                        and attr is not base_class
                    ):
                        candidate_classes.append((attr_name, attr))

                if len(candidate_classes) == 0:
                    raise ValueError(
                        f"No class found in code that inherits from {base_class.__name__}"
                    )
                elif len(candidate_classes) == 1:
                    class_name = candidate_classes[0][0]
                else:
                    # 处理工具调用。
                    preferred = []
                    for name, cls in candidate_classes:
                        try:
                            source = inspect.getsource(cls)
                            if (
                                "@AGENT" in source
                                or "@TOOL" in source
                                or "@PROMPT" in source
                                or name.endswith(("Agent", "Tool", "Prompt"))
                            ):
                                preferred.append((name, cls))
                        except (OSError, TypeError):
                            if name.endswith(("Agent", "Tool", "Prompt")):
                                preferred.append((name, cls))

                    if preferred:
                        class_name = preferred[0][0]
                    else:
                        # 说明相关实现细节。
                        class_name = candidate_classes[0][0]
            else:
                # 说明相关实现细节。
                class_name = self.extract_class_name_from_code(code)
                if not class_name:
                    raise ValueError(
                        "Cannot determine class name from code. Please provide class_name or base_class."
                    )

        # 说明相关实现细节。
        if not hasattr(module, class_name):
            raise ValueError(f"Class {class_name} not found in the provided code")

        cls = getattr(module, class_name)

        # 校验输入与当前状态。
        if base_class is not None and not issubclass(cls, base_class):
            raise ValueError(
                f"Class {class_name} is not a subclass of {base_class.__name__}"
            )

        return cls

    def load_function(
        self,
        code: str,
        function_name: str | None = None,
        module_name: str | None = None,
    ) -> Any:
        """加载与 `load_function` 对应的数据或状态。"""
        # 说明相关实现细节。
        if function_name is None:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        function_name = node.name
                        break
            except (SyntaxError, TypeError):
                function_name = None

            if not function_name:
                raise ValueError(
                    "Cannot determine function name from code. Please provide function_name."
                )

        # 加载所需数据。
        if module_name is None:
            module_name = self.load_code(code)
        else:
            if module_name not in self._loaded_modules:
                self.load_code(code, module_name)

        # 说明相关实现细节。
        module = self._loaded_modules[module_name]

        # 说明相关实现细节。
        if not hasattr(module, function_name):
            raise ValueError(f"Function {function_name} not found in the provided code")

        return getattr(module, function_name)

    def get_module(self, module_name: str) -> Any | None:
        """获取与 `get_module` 对应的数据或状态。"""
        return self._loaded_modules.get(module_name)

    def get_class_string(self, cls: type) -> str | None:
        """获取与 `get_class_string` 对应的数据或状态。"""
        if not isinstance(cls, type):
            return None

        # 说明相关实现细节。
        module_name = getattr(cls, "__module__", None)
        if not module_name:
            return None

        # 说明相关实现细节。
        class_name = getattr(cls, "__name__", None)
        if not class_name:
            return None

        return f"<{module_name}.{class_name}>"

    def list_loaded_modules(self) -> list[str]:
        """列出与 `list_loaded_modules` 对应的数据或状态。"""
        return list(self._loaded_modules.keys())

    def get_parameters(self, object: type["T"] | Callable) -> dict[str, Any]:
        """获取与 `get_parameters` 对应的数据或状态。"""
        try:
            if isinstance(object, type):
                signature = inspect.signature(object.__call__)
                hints = get_type_hints(object.__call__)
                docstring = inspect.getdoc(object.__call__) or ""
            elif isinstance(object, Callable):
                signature = inspect.signature(object)
                hints = get_type_hints(object)
                docstring = inspect.getdoc(object) or ""
        except Exception as e:
            raise ValueError(f"Failed to get parameters for {object}: {e}")

        # 说明相关实现细节。
        doc_descriptions = self.parse_docstring_descriptions(docstring)

        properties = {}
        required = []

        for name, param in signature.parameters.items():
            if name == "self":
                continue

            # 处理输入参数。
            if name == "input" and len(signature.parameters) == 2:  # 处理输入参数。
                continue

            # 处理输入参数。
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # 说明相关实现细节。
            annotation = hints.get(name, param.annotation)
            json_type, python_type = self.annotation_to_types(annotation)

            # 说明相关实现细节。
            is_required = param.default is inspect._empty

            # 创建所需对象。
            schema: dict[str, Any] = {
                "type": json_type,
                "description": doc_descriptions.get(name, ""),
            }
            schema[PYTHON_TYPE_FIELD] = python_type

            if not is_required:
                schema["default"] = param.default

            properties[name] = schema
            if is_required:
                required.append(name)

        if not properties:
            return self.default_parameters_schema()

        result: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            result["required"] = required
        return result

    def serialize_args_schema(
        self, args_schema: type[BaseModel]
    ) -> dict[str, Any] | None:
        """序列化与 `serialize_args_schema` 对应的数据或状态。"""
        try:
            schema_info = {"class_name": args_schema.__name__, "fields": {}}

            # 转换并规范化数据。
            for field_name, field_info in args_schema.model_fields.items():
                field_data = {
                    "type": str(field_info.annotation)
                    if hasattr(field_info, "annotation")
                    else "Any",
                    "required": field_info.is_required()
                    if hasattr(field_info, "is_required")
                    else True,
                }

                # 说明相关实现细节。
                if hasattr(field_info, "description") and field_info.description:
                    field_data["description"] = field_info.description

                # 说明相关实现细节。
                if hasattr(field_info, "default") and field_info.default is not ...:
                    if field_info.default is not None:
                        # 转换并规范化数据。
                        try:
                            json.dumps(field_info.default)
                            field_data["default"] = field_info.default
                        except (TypeError, ValueError):
                            field_data["default"] = None
                    else:
                        field_data["default"] = None

                schema_info["fields"][field_name] = field_data

            return schema_info
        except Exception as e:
            raise ValueError(
                f"Failed to serialize args_schema {args_schema.__name__}: {e}"
            )

    def deserialize_args_schema(
        self, schema_info: dict[str, Any]
    ) -> type[BaseModel] | None:
        """反序列化与 `deserialize_args_schema` 对应的数据或状态。"""
        try:
            class_name = schema_info.get("class_name")
            fields_info = schema_info.get("fields", {})

            if not class_name:
                return None

            # 创建所需对象。
            field_definitions = {}
            for field_name, field_data in fields_info.items():
                # 转换并规范化数据。
                type_str = field_data.get("type", "Any")
                python_type = self.parse_type_string(type_str)

                # 说明相关实现细节。
                default_value = field_data.get("default")
                is_required = field_data.get("required", True)

                if default_value is None and not is_required:
                    # 说明相关实现细节。
                    python_type = python_type | None if python_type != Any else Any
                    default_value = None
                elif default_value is None and is_required:
                    default_value = ...  # 说明相关实现细节。

                # 创建所需对象。
                description = field_data.get("description", "")
                if description:
                    field_definitions[field_name] = (
                        python_type,
                        Field(default=default_value, description=description),
                    )
                else:
                    field_definitions[field_name] = (python_type, default_value)

            # 创建所需对象。
            model = create_model(
                class_name,
                __config__=ConfigDict(arbitrary_types_allowed=True, extra="allow"),
                **field_definitions,
            )

            return model
        except Exception as e:
            raise ValueError(f"Failed to deserialize args_schema: {e}")


# 说明相关实现细节。
dynamic_manager = DynamicModuleManager()
