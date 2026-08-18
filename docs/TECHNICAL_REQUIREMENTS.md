# 技术需求

## 一、运行时

- 使用 Python 3.11 或更高版本。
- 采用异步优先的执行方式，并显式支持取消和超时。
- 使用 Pydantic v2 定义请求、响应、事件和配置模型。
- 导入模块时不得发起网络请求、修改文件系统或初始化模型提供方。
- 通过应用容器进行依赖注入。
- 智能体、工具、提示词、记忆和环境共享统一的资源生命周期抽象。

## 二、模型提供方

运行时必须支持独立配置模型提供方，智能体业务逻辑中不得出现按厂商名称分支的判断。

必须支持：

- OpenAI
- OpenRouter
- 阿里云百炼／通义千问
- DeepSeek

可选支持：

- Anthropic
- Google Gemini

每个模型必须声明能力，不得仅根据模型名称推断：

- 流式输出
- 结构化输出
- 工具调用
- 推理模式
- 视觉与文档输入
- 上下文窗口和输出限制
- 词元与成本统计

提供方必须通过环境变量读取密钥和服务地址。千问使用 `DASHSCOPE_API_KEY` 与 `QWEN_BASE_URL`；DeepSeek 使用 `DEEPSEEK_API_KEY` 与 `DEEPSEEK_API_BASE`。

当前支持的配置结构：

```yaml
models:
  default: qwen/qwen-plus
  fallback:
    - deepseek/deepseek-v4-flash
  embedding: qwen/text-embedding-v4

providers:
  qwen:
    adapter: openai_compatible
    api_key_env: DASHSCOPE_API_KEY
    base_url_env: QWEN_BASE_URL
  deepseek:
    adapter: openai_compatible
    api_key_env: DEEPSEEK_API_KEY
    base_url_env: DEEPSEEK_API_BASE
```

不得使用已经停用的 DeepSeek 别名 `deepseek-chat` 和 `deepseek-reasoner`。模型标识必须保留在配置中，因为提供方的模型目录会持续变化。

聊天模型与 Embedding 模型必须独立配置。DeepSeek 只作为聊天模型使用时，RAG 可继续选择 Qwen `text-embedding-v4` 或 OpenAI Embedding。

## 三、研究流水线

- 明确定义计划、收集、分析、验证和报告状态。
- 保存来源地址、标题、获取时间、内容哈希和引用片段链路。
- 支持重复来源检测和冲突证据说明。
- 支持重试策略和部分失败恢复。
- 提供确定性的虚拟模型与虚拟工具，供离线测试使用。

## 四、安全

- 将用户附件、网页内容、记忆和工具输出视为不可信数据。
- 不可信文本进入模型上下文前必须经过提示注入扫描。
- 指令和证据必须保存在不同的类型化消息字段中。
- 文件操作必须限制在指定工作区根目录内。
- 拒绝绝对路径逃逸、目录穿越和不安全的符号链接解析。
- 高风险工具必须经过能力授权和人工审批。
- 未来的 Python 或命令行能力必须运行在隔离工作进程或容器中，禁止使用不受限的 `shell=True` 或进程内 `exec`。
- 日志与执行轨迹必须隐藏密钥、Cookie、授权头和个人数据。

## 五、可观测性与评测

- 为运行、步骤、模型调用、工具调用、引用和安全决策生成结构化事件。
- 统计成功率、引用正确率、P50／P95 延迟、词元、成本、工具失败率和注入拦截率。
- 支持导出与 OpenTelemetry 兼容的执行轨迹。
- 建立版本化评测集，覆盖正常研究、提供方故障、恶意文档和冲突来源。

## 六、交付质量

- 使用 `pyproject.toml` 划分核心、模型提供方、研究、界面和开发依赖组。
- 使用 Ruff、mypy、pytest、pytest-asyncio 和覆盖率门禁。
- 使用 GitHub Actions 执行静态检查、单元测试、安全检查和离线冒烟测试。
- 首先提供容器化命令行演示；运行时稳定后再增加 FastAPI 与轻量网页界面。
