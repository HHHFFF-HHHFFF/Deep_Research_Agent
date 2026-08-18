# 深度研究智能体

一个本地运行的个人深度研究助手。用户输入研究主题后，工具型 Agent 会搜索和分析网页资料，并生成中文 Markdown 研究报告。

项目重点展示 Agent 工具调用、Qwen／DeepSeek 模型路由、网页研究和 FAISS 本地 RAG。前端采用轻量级 Streamlit 单页方案，不建设复杂平台。

## 项目定位

这是一个可在面试中完整讲清楚的个人项目：

- 单个用户在本地运行
- 输入一个研究主题
- 可选择 Qwen 或 DeepSeek
- Agent 调用搜索、抓取、分析和报告工具
- 页面展示并下载最终报告
- 可选上传本地文档并复用 FAISS 检索

项目明确不做 FastAPI、Next.js、数据库、Redis、任务队列、用户系统和多租户。

## 当前状态

已完成：

- 动态研究主题输入
- Qwen／DeepSeek 聊天模型切换
- 独立 Embedding 配置和备用模型降级
- 工具型 Agent、网页研究、报告生成和 FAISS 基础模块
- P1-M1 全仓 Ruff 规范清理
- P1-M2a 至 P1-M2d 的高价值类型修复
- 命令行研究入口

下一步是从现有命令行提取一个轻量异步调用函数，然后增加单页 Streamlit 界面。原应用服务层和平台化计划已取消。

## 技术栈

| 技术 | 用途 |
| --- | --- |
| Python 3.10、AsyncIO | Agent 主程序和异步工具调用 |
| Streamlit | 本地单页用户界面 |
| Pydantic v2 | 研究主题与文件输入校验 |
| MMEngine Config、python-dotenv | 配置合并和本地密钥读取 |
| Qwen、DeepSeek | 聊天模型与工具调用 |
| OpenAI 兼容接口 | 统一连接不同模型提供方 |
| Crawl4AI、DDGS、HTTPX | 网页搜索、抓取和解析 |
| FAISS、Qwen Embedding | 本地文档向量检索 |
| pytest、Ruff、mypy | 离线测试、规范与类型检查 |

## 核心流程

```text
研究主题／可选文档
        ↓
工具型 Agent
        ↓
搜索 → 抓取 → 分析 → 报告
        ↓
Markdown 研究报告
```

Streamlit 只负责收集输入和展示结果，现有 Agent 与研究工具仍是项目主体。

## 项目结构

```text
configs/       研究场景与模型配置
docs/          项目范围、技术需求和开发计划
examples/      当前命令行入口
src/
  application/ 研究输入校验
  agent/       工具型 Agent 执行循环
  model/       Qwen／DeepSeek 等模型适配与降级
  tool/        搜索、文档分析和报告工具
  environment/ 文件系统与 FAISS 检索
  memory/      研究过程记忆
  prompt/      工具调用提示模板
tests/         离线测试
```

计划新增的界面代码只保留：

```text
streamlit_app.py       单页界面
src/research_runner.py 命令行与界面共用的轻量研究调用函数
```

## 配置模型

复制 `.env.template` 为 `.env`，密钥只保存在本地，不得提交到 GitHub。

默认使用 Qwen：

```env
MODEL_PROVIDER=qwen
MODEL_NAME=qwen-plus
EMBEDDING_PROVIDER=qwen
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=你的密钥
```

切换到 DeepSeek 时，聊天模型使用 DeepSeek，FAISS 仍可使用 Qwen Embedding：

```env
MODEL_PROVIDER=deepseek
MODEL_NAME=deepseek-v4-flash
DEEPSEEK_API_KEY=你的密钥

EMBEDDING_PROVIDER=qwen
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=你的密钥
```

模型标识可能随提供方更新，请以实际账号当前可用模型为准。

## 当前运行方式

在项目解释器环境中执行：

```bash
python examples/run_tool_calling_agent.py --task "调研 RAG 系统中的幻觉问题，并生成中文报告。"
```

临时切换聊天模型：

```bash
python examples/run_tool_calling_agent.py --task "研究主题" --provider deepseek --model deepseek-v4-flash
```

省略 `--task` 时，终端会提示输入研究主题。

Streamlit 界面将在下一阶段完成，届时使用：

```bash
streamlit run streamlit_app.py
```

## 设计文档

- [项目范围](docs/PROJECT_SCOPE.md)
- [轻量开发计划](docs/DEVELOPMENT_PLAN.md)
- [技术需求](docs/TECHNICAL_REQUIREMENTS.md)
- [许可证中文说明](LICENSE_zh.md)

## 范围原则

后续功能必须直接服务于“输入主题并生成研究报告”。如果一个需求需要数据库、服务端 API、队列、多用户或多个新业务模块，默认不加入本项目。
