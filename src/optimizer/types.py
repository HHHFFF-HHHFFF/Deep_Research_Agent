"""提供数据类型相关实现。"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.session import SessionContext
from collections import defaultdict
from functools import partial
from typing import Union

from jinja2 import Environment, Template, TemplateError, meta


class Variable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="The name of the variable.")
    type: str = Field(description="The type of the variable.")
    description: str = Field(description="The description of the variable.")
    require_grad: bool = Field(
        default=False, description="Whether the variable requires gradient."
    )
    template: str | None = Field(
        default=None, description="The template of the variable."
    )
    variables: Union[dict[str, "Variable"], "Variable", Any] | None = Field(
        default=None,
        description="The elements of the variable. Can be a dict (keyed by name), single Variable, or direct value.",
    )

    # 说明相关实现细节。
    gradients: set["Variable"] = Field(
        default_factory=set, description="Text gradients for this variable."
    )
    gradients_context: dict["Variable", str] = Field(
        default_factory=lambda: defaultdict(lambda: None),
        description="Context for gradients.",
    )
    grad_fn: Any | None = Field(
        default=None, description="Gradient function for backward pass."
    )
    predecessors: set["Variable"] = Field(
        default_factory=set, description="Predecessor variables in computation graph."
    )
    reduce_meta: list[dict] = Field(
        default_factory=list, description="Metadata for gradient reduction."
    )

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Variable":
        """实现 `from_dict` 的业务逻辑。"""
        subvars = data.get("variables")
        if isinstance(subvars, dict):
            # 转换并规范化数据。
            subvars = {
                k: cls.from_dict(v) if isinstance(v, dict) and "name" in v else v
                for k, v in subvars.items()
            }
        elif subvars is not None and not isinstance(subvars, dict):
            # 说明相关实现细节。
            pass
        return cls(
            name=data["name"],
            type=data.get("type", ""),
            description=data.get("description", ""),
            require_grad=data.get("require_grad", False),
            template=data.get("template"),
            variables=subvars,
        )

    def render(self, modules: dict[str, Any]) -> str:
        """实现 `render` 的业务逻辑。"""
        if self.template is None:
            return ""

        env = Environment()
        ast = env.parse(self.template)
        vars_used = meta.find_undeclared_variables(ast)
        ctx = dict(modules)

        for var in vars_used:
            if var in modules:
                val = modules[var]
                if isinstance(val, str):
                    # 说明相关实现细节。
                    try:
                        temp_template = Template(val)
                        ctx[var] = temp_template.render(**ctx)
                    except (TemplateError, TypeError, ValueError):
                        # 处理异常情况。
                        ctx[var] = val
                else:
                    ctx[var] = val

        return Template(self.template).render(**ctx)

    def get_modules(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """获取与 `get_modules` 对应的数据或状态。"""
        result: dict[str, Any] = {}
        ctx = dict(context or {})

        # 创建所需对象。
        if isinstance(self.variables, dict):
            # 转换并规范化数据。
            for child in self.variables.values():
                if isinstance(child, Variable):
                    child_modules = child.get_modules(ctx)
                    result.update(child_modules)
                    ctx.update(child_modules)
        elif isinstance(self.variables, Variable):
            child_modules = self.variables.get_modules(ctx)
            result.update(child_modules)
            ctx.update(child_modules)
        elif self.variables is not None:
            # 说明相关实现细节。
            result[self.name] = self.variables
            ctx[self.name] = self.variables

        # 说明相关实现细节。
        if self.template is not None:
            try:
                rendered = Template(self.template).render(**ctx)
                result[self.name] = rendered
                ctx[self.name] = rendered
            except Exception:
                # 处理异常情况。
                result[self.name] = self.template
                ctx[self.name] = self.template

        return result

    def get_value(self) -> str:
        """获取与 `get_value` 对应的数据或状态。"""
        if self.template is None:
            # 组装并返回结果。
            if isinstance(self.variables, dict):
                # 转换并规范化数据。
                return " ".join(
                    [
                        child.get_value()
                        for child in self.variables.values()
                        if isinstance(child, Variable)
                    ]
                )
            elif isinstance(self.variables, Variable):
                return self.variables.get_value()
            elif self.variables is not None:
                return str(self.variables)
            else:
                return ""

        # 说明相关实现细节。
        modules = self.get_modules()
        return self.render(modules)

    def __repr__(self):
        return f"Variable(name={self.name}, type={self.type}, value={self.get_value()}, role={self.description}, grads={len(self.gradients)})"

    def __str__(self):
        return self.get_value()

    def __add__(self, to_add):
        """实现 `__add__` 的业务逻辑。"""
        if isinstance(to_add, Variable):
            # 创建所需对象。
            result = Variable(
                name=f"{self.name}_plus_{to_add.name}",
                type="computed",
                description=f"{self.description} and {to_add.description}",
                require_grad=(self.require_grad or to_add.require_grad),
                template="{{var1}} {{var2}}",  # 说明相关实现细节。
                variables=[self, to_add],
                predecessors={self, to_add},
            )
            # 更新相关状态。
            result.set_grad_fn(
                partial(
                    self._backward_idempotent,
                    variables=[self, to_add],
                    summation=result,
                )
            )
            return result
        else:
            return to_add.__add__(self)

    def set_grad_fn(self, grad_fn):
        """设置与 `set_grad_fn` 对应的数据或状态。"""
        self.grad_fn = grad_fn

    def get_grad_fn(self):
        """获取与 `get_grad_fn` 对应的数据或状态。"""
        return self.grad_fn

    def reset_gradients(self):
        """实现 `reset_gradients` 的业务逻辑。"""
        self.gradients = set()
        self.gradients_context = defaultdict(lambda: None)
        self.reduce_meta = []

        # 更新相关状态。
        if isinstance(self.variables, dict):
            # 转换并规范化数据。
            for child in self.variables.values():
                if isinstance(child, Variable):
                    child.reset_gradients()
        elif isinstance(self.variables, Variable):
            self.variables.reset_gradients()

    def get_gradient_text(self) -> str:
        """获取与 `get_gradient_text` 对应的数据或状态。"""
        return "\n".join([g.get_value() for g in self.gradients])

    def backward(self, engine: Any = None):
        """实现 `backward` 的业务逻辑。"""
        # 说明相关实现细节。
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for predecessor in v.predecessors:
                    build_topo(predecessor)
                topo.append(v)

        build_topo(self)

        # 说明相关实现细节。
        self.gradients = set()
        for v in reversed(topo):
            if v.require_grad:
                v.gradients = self._check_and_reduce_gradients(v, engine)
                if v.get_grad_fn() is not None:
                    v.grad_fn(backward_engine=engine)

    def _check_and_reduce_gradients(
        self, variable: "Variable", backward_engine=None
    ) -> set["Variable"]:
        """实现 `_check_and_reduce_gradients` 的业务逻辑。"""
        if variable.reduce_meta == []:
            return variable.gradients
        if variable.get_gradient_text() == "":
            return variable.gradients

        if len(variable.gradients) == 1:
            return variable.gradients

        # 说明相关实现细节。
        id_to_gradient_set = defaultdict(set)
        id_to_op = {}

        for gradient in variable.gradients:
            for reduce_item in gradient.reduce_meta:
                id_to_gradient_set[reduce_item["id"]].add(gradient)
                id_to_op[reduce_item["id"]] = reduce_item["op"]

        new_gradients = set()
        for group_id, gradients in id_to_gradient_set.items():
            new_gradients.add(id_to_op[group_id](gradients, backward_engine))

        return new_gradients

    def _backward_idempotent(
        self, variables: list["Variable"], summation: "Variable", backward_engine=None
    ):
        """实现 `_backward_idempotent` 的业务逻辑。"""
        summation_gradients = summation.get_gradient_text()
        for variable in variables:
            if summation_gradients == "":
                variable_gradient_value = ""
            else:
                variable_gradient_value = f"Here is the combined feedback for this specific {variable.description} and other variables: {summation_gradients}."

            var_gradients = Variable(
                name=f"gradient_for_{variable.name}",
                type="gradient",
                description=f"feedback to {variable.description}",
                require_grad=False,
                variables=variable_gradient_value,
            )
            variable.gradients.add(var_gradients)

            if summation.reduce_meta != []:
                var_gradients.reduce_meta.extend(summation.reduce_meta)
                variable.reduce_meta.extend(summation.reduce_meta)

    def generate_graph(self, print_gradients: bool = False):
        """生成与 `generate_graph` 对应的数据或状态。"""
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError(
                "Please install graphviz to visualize the computation graphs."
            )

        def wrap_text(text, width=40):
            words = text.split()
            wrapped_text = ""
            line = ""
            for word in words:
                if len(line) + len(word) + 1 > width:
                    wrapped_text += line + "<br/>"
                    line = word
                else:
                    if line:
                        line += " "
                    line += word
            wrapped_text += line
            return wrapped_text

        def wrap_and_escape(text, width=40):
            return wrap_text(text.replace("<", "&lt;").replace(">", "&gt;"), width)

        # 创建所需对象。
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for predecessor in v.predecessors:
                    build_topo(predecessor)
                topo.append(v)

        build_topo(self)

        graph = Digraph(comment=f"Computation Graph starting from {self.description}")
        graph.attr(rankdir="TB")
        graph.attr(ranksep="0.2")
        graph.attr(bgcolor="lightgrey")
        graph.attr(fontsize="7.5")

        for v in reversed(topo):
            label_color = "darkblue"

            node_label = (
                f"<b><font color='{label_color}'>Name: </font></b> {wrap_and_escape(v.name)}"
                f"<br/><b><font color='{label_color}'>Description: </font></b> {wrap_and_escape(v.description)}"
                f"<br/><b><font color='{label_color}'>Value: </font></b> {wrap_and_escape(v.get_value())}"
            )

            if v.grad_fn is not None:
                node_label += f"<br/><b><font color='{label_color}'>Grad Fn: </font></b> {wrap_and_escape(str(v.grad_fn))}"

            if print_gradients:
                node_label += f"<br/><b><font color='{label_color}'>Gradients: </font></b> {wrap_and_escape(v.get_gradient_text())}"

            graph.node(
                str(id(v)),
                label=f"<{node_label}>",
                shape="rectangle",
                style="filled",
                fillcolor="lavender",
                fontsize="8",
                fontname="Arial",
                margin="0.1",
                pad="0.1",
                width="1.2",
            )

            for predecessor in v.predecessors:
                graph.edge(str(id(predecessor)), str(id(v)))

        return graph

    def get_all_variables(self) -> list["Variable"]:
        """获取与 `get_all_variables` 对应的数据或状态。"""
        all_vars = [self]

        if isinstance(self.variables, dict):
            # 转换并规范化数据。
            for child in self.variables.values():
                if isinstance(child, Variable):
                    all_vars.extend(child.get_all_variables())
        elif isinstance(self.variables, Variable):
            all_vars.extend(self.variables.get_all_variables())

        return all_vars

    def get_trainable_variables(self) -> dict[str, "Variable"]:
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        trainable_vars: dict[str, Variable] = {}

        # 校验输入与当前状态。
        if isinstance(self.variables, dict) and len(self.variables) > 0:
            # 组装并返回结果。
            for key, child in self.variables.items():
                if isinstance(child, Variable) and child.require_grad:
                    trainable_vars[key] = child
        elif not isinstance(self.variables, (dict, Variable)) and self.require_grad:
            trainable_vars[self.name] = self

        return trainable_vars


class Optimizer(BaseModel):
    """定义 `Optimizer`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        workdir: str,
        model_name: str | None = None,
        prompt_name: str | None = None,
        memory_name: str | None = None,
        max_steps: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 更新相关状态。
        self.workdir = workdir

        # 更新相关状态。
        self.prompt_name = prompt_name
        self.memory_name = memory_name
        self.model_name = model_name

        # 初始化相关状态。
        self.max_steps = max_steps if max_steps > 0 else int(1e8)

        # 初始化相关状态。
        self.optimizable_vars = []
        self.var_mapping = {}
        self.prompt_mapping = None

    async def get_trainable_variables(self):
        """获取与 `get_trainable_variables` 对应的数据或状态。"""
        raise NotImplementedError(
            f"``get_trainable_variables`` function for {type(self).__name__} is not implemented!"
        )

    async def set_trainable_variables(self, variables: list["Variable"]):
        """设置与 `set_trainable_variables` 对应的数据或状态。"""
        raise NotImplementedError(
            f"``set_trainable_variables`` function for {type(self).__name__} is not implemented!"
        )

    async def optimize(
        self,
        task: str,
        files: list[str] | None = None,
        ctx: "SessionContext" = None,
        **kwargs,
    ):
        """实现 `optimize` 的业务逻辑。"""
        raise NotImplementedError(
            f"``optimize`` function for {type(self).__name__} is not implemented!"
        )

    def close(self):
        """关闭资源并完成清理。"""
        raise NotImplementedError(
            f"``close`` function for {type(self).__name__} is not implemented!"
        )
