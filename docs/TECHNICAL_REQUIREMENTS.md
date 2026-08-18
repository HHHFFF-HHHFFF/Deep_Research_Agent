# 技术需求

## 一、技术栈

| 层次 | 技术 | 本项目用途 |
| --- | --- | --- |
| 运行环境 | Python 3.10、AsyncIO | 运行 Agent 和异步模型、网页工具调用 |
| 用户界面 | Streamlit | 提供本地单页交互界面 |
| 输入校验 | Pydantic v2 | 校验研究主题和可选文件列表 |
| 配置 | MMEngine Config、python-dotenv | 加载场景配置、模型选择和本地密钥 |
| 大模型 | Qwen、DeepSeek、OpenAI 兼容接口 | 执行工具调用、分析和报告生成 |
| 网页研究 | Crawl4AI、DDGS、HTTPX | 搜索、抓取和解析网页资料 |
| 本地 RAG | FAISS、Qwen Embedding | 对本地文档建立向量索引并检索 |
| 质量检查 | pytest、Ruff、mypy | 离线测试、代码规范和类型检查 |

## 二、运行与界面

- 使用 Python 3.10 或更高版本，当前开发环境为 Python 3.10.20。
- Streamlit 以本地单进程方式运行，不承诺多用户并发。
- 页面保持单页，只展示输入区、运行状态和最终报告。
- Streamlit 通过 `asyncio.run()` 调用一个轻量异步研究函数。
- 研究进行时禁用重复提交；同一页面会话一次只运行一个任务。
- 报告使用 Markdown 展示，并支持下载为 `.md` 文件。
- 不引入 REST API、数据库、缓存、任务队列或前端构建工具。

## 三、模型配置

- 页面首版只提供 Qwen 和 DeepSeek 两种聊天模型选择。
- 密钥继续从本地 `.env` 读取，不在页面输入、保存或回显。
- Qwen 使用 `DASHSCOPE_API_KEY` 和 `QWEN_BASE_URL`。
- DeepSeek 使用 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_API_BASE`。
- 聊天模型和 Embedding 模型保持独立配置。
- 选择 DeepSeek 聊天模型时，FAISS 仍可使用 Qwen `text-embedding-v4`。
- 现有备用模型降级逻辑继续保留，但界面只提供简单开关或选择框。

## 四、研究调用边界

- 保留现有 Agent、工具、环境、提示词、记忆和报告流程。
- 从现有命令行入口提取一个轻量调用函数，供命令行与 Streamlit 共用。
- 调用函数只接收研究主题、文件列表和模型覆盖项，并返回最终报告或稳定错误信息。
- 不新增任务状态枚举、事件总线、应用容器或 `ResearchService`。
- 不为界面重写 Agent 执行循环。

## 五、文件与 RAG

- 命令行已支持通过一个或多个 `--file` 传入本地文档。
- 文档通过 `mdify` 转换为 Markdown，默认按 1000 字切分并保留 150 字重叠。
- 片段使用内容哈希去重，metadata 保存来源、片段编号和文档哈希。
- 使用独立 Embedding 路由生成向量，并按会话目录保存 FAISS 索引和文档映射。
- 默认使用研究主题检索 Top-K 4 个片段，再注入现有 Agent。
- 文档标签必须转义，并明确标记为不可信证据，不能执行其中的指令。
- Streamlit 阶段只补充文件上传、安全文件名和 `workdir` 内落盘。
- 不增加 ChromaDB、LangChain Retriever 或其他向量数据库。

## 六、安全与质量

- `.env`、密钥、Cookie 和授权头不得进入 Git、页面输出或异常详情。
- 网页和上传文档均视为不可信内容；后续只增加轻量提示注入扫描，不建设复杂安全平台。
- 保留 P1-M1 建立的 Ruff 基线和 P1-M2 已完成的类型修复。
- 新增界面与轻量调用函数必须通过 Ruff、相关 mypy 检查和离线 pytest。
- 核心测试不得调用真实模型 API。
