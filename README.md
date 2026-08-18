# 深度研究智能体

一个面向个人作品集的安全、可观测、可评测人工智能深度研究助手。

用户输入研究问题及可选文档后，系统将完成规划、检索、分析、交叉验证和报告生成，并记录来源、工具调用、耗时、词元、成本与安全事件。

## 当前状态

本项目由个人从零规划并持续搭建，目标是完整呈现一个深度研究智能体从需求分析、架构设计到工程实现和评测交付的全过程。

当前已经完成基础智能体框架、Qwen/DeepSeek 模型路由、独立 Embedding 配置、动态研究请求和统一 `ResearchService`。下一步将让命令行接入该应用服务，再补全研究闭环、安全与可观测性，最后交付 FastAPI + Next.js 单租户平台。

## 目标能力

- 深度研究闭环：计划、检索、阅读、验证、引用和报告
- 多模型接入：OpenAI、OpenRouter、通义千问、DeepSeek，可选 Anthropic 和 Gemini
- 能力感知路由、失败降级、词元与成本统计
- 提示注入检测、来源隔离、工作区文件沙箱和危险操作审批
- 执行轨迹、结构化日志和可复现评测
- 命令行优先，运行时稳定后再提供 FastAPI 与 Next.js 单租户平台界面

## 项目结构

```text
configs/       单一研究场景配置
docs/          项目范围与技术需求
examples/      单一研究智能体入口
src/
  application/ 研究请求与应用服务层
  agent/       智能体执行循环
  model/       当前模型适配层
  tool/        研究、文档和报告工具
  environment/ 文件系统与本地向量检索
  memory/      研究过程记忆
  prompt/      工具调用提示模板
  skill/       扩展契约
  tracer/      执行轨迹
  version/     组件版本记录
```

## 设计文档

- [项目范围](docs/PROJECT_SCOPE.md)
- [开发计划与当前进度](docs/DEVELOPMENT_PLAN.md)
- [技术需求](docs/TECHNICAL_REQUIREMENTS.md)
- [许可证中文说明](LICENSE_zh.md)

## 当前运行入口

### 1. 配置模型

复制 `.env.template` 为 `.env`。默认使用 Qwen 生成答案，并使用独立的 Qwen Embedding 为 FAISS 提供向量：

```env
MODEL_PROVIDER=qwen
MODEL_NAME=qwen-plus
EMBEDDING_PROVIDER=qwen
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=你的密钥
```

切换到 DeepSeek 时只需要修改聊天模型；DeepSeek 当前不提供本项目所需的 Embedding 接口，因此向量模型仍可使用 Qwen：

```env
MODEL_PROVIDER=deepseek
MODEL_NAME=deepseek-v4-flash
DEEPSEEK_API_KEY=你的密钥

EMBEDDING_PROVIDER=qwen
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=你的密钥
```

也可以配置自动降级：

```env
MODEL_PROVIDER=qwen
MODEL_NAME=qwen-plus
MODEL_FALLBACKS=deepseek/deepseek-v4-flash
```

密钥只保存在本地 `.env` 中，不得提交到 GitHub。

### 2. 启动项目

```bash
python examples/run_tool_calling_agent.py --task "调研 RAG 系统中的幻觉问题，并生成带来源的中文报告。"
```

如果省略 `--task`，程序会在终端中提示输入研究主题。研究主题不再写死在源码中。

也可以通过命令行临时切换聊天模型：

```bash
python examples/run_tool_calling_agent.py --task "研究主题" --provider deepseek --model deepseek-v4-flash
```

## 模型接入架构

- `ModelRuntimeSettings`：统一读取默认模型、备用模型、向量模型、超时和重试配置
- `ChatOpenAICompatible`：使用同一适配器连接 Qwen、DeepSeek、OpenAI 和 OpenRouter
- `ModelManager`：只初始化本次实际使用的模型，并在失败响应或异常时按顺序降级
- `aembedding()`：为 FAISS 提供独立的 Embedding 路由，避免与聊天模型绑定

项目目前处于持续开发阶段，该入口将在后续迭代中升级为正式命令行工具，并逐步补充自动化测试、评测数据和部署方案。
