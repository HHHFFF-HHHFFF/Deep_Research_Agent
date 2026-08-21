import {
  ArrowRightOutlined,
  CheckCircleFilled,
  FileSearchOutlined,
  FileTextOutlined,
  GlobalOutlined,
  InboxOutlined,
  SafetyCertificateOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  ConfigProvider,
  Form,
  Input,
  Radio,
  Tag,
  Typography,
  Upload,
  type UploadFile,
  type UploadProps,
} from "antd";
import zhCN from "antd/locale/zh_CN";
import { useState } from "react";

import { ApiClientError, createResearchTask, type ModelProvider } from "./api";
import { RecentTasks } from "./components/RecentTasks";
import { TaskWorkspace } from "./components/TaskWorkspace";
import { useResearchWorkspace } from "./useResearchWorkspace";
import "./App.css";

const { Dragger } = Upload;
const { Text, Title, Paragraph } = Typography;

const MODEL_OPTIONS: Record<
  ModelProvider,
  { title: string; modelId: string; description: string; mark: string }
> = {
  qwen: {
    title: "Qwen",
    modelId: "qwen3-max",
    description: "适合中文研究与工具调用",
    mark: "Q",
  },
  deepseek: {
    title: "DeepSeek",
    modelId: "deepseek-v4-flash",
    description: "侧重推理与内容分析",
    mark: "D",
  },
};

const ALLOWED_EXTENSIONS = [".md", ".txt", ".pdf", ".docx"];
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_FILE_COUNT = 5;
const MAX_TOTAL_FILE_BYTES = 25 * 1024 * 1024;

interface ResearchFormValues {
  task: string;
  provider: ModelProvider;
}

function fileExtension(name: string): string {
  const dotIndex = name.lastIndexOf(".");
  return dotIndex >= 0 ? name.slice(dotIndex).toLowerCase() : "";
}

function getNativeFiles(fileList: UploadFile[]): File[] {
  return fileList.flatMap((file) => (file.originFileObj ? [file.originFileObj] : []));
}

function App() {
  const [form] = Form.useForm<ResearchFormValues>();
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [uploadErrorMessage, setUploadErrorMessage] = useState<string | null>(null);
  const workspaceState = useResearchWorkspace();
  const selectedProvider = Form.useWatch("provider", form) ?? "qwen";
  const controlsDisabled = submitting || workspaceState.hasActiveTask;

  const uploadProps: UploadProps = {
    multiple: true,
    accept: ALLOWED_EXTENSIONS.join(","),
    maxCount: MAX_FILE_COUNT,
    fileList,
    disabled: controlsDisabled,
    beforeUpload: (file) => {
      if (!ALLOWED_EXTENSIONS.includes(fileExtension(file.name))) {
        setUploadErrorMessage("仅支持 Markdown、TXT、PDF 和 DOCX 文件");
        return Upload.LIST_IGNORE;
      }
      if (file.size > MAX_FILE_BYTES) {
        setUploadErrorMessage("单个文件不能超过 10 MB");
        return Upload.LIST_IGNORE;
      }
      setUploadErrorMessage(null);
      return false;
    },
    onChange: ({ fileList: nextFileList }) => {
      setFileList(nextFileList.slice(-MAX_FILE_COUNT));
      setErrorMessage(null);
      setUploadErrorMessage(null);
    },
  };

  const submitResearch = async (values: ResearchFormValues) => {
    if (submitting || workspaceState.hasActiveTask) return;
    const nativeFiles = getNativeFiles(fileList);
    if (nativeFiles.reduce((total, file) => total + file.size, 0) > MAX_TOTAL_FILE_BYTES) {
      setUploadErrorMessage("单次研究的资料总量不能超过 25 MB");
      return;
    }
    setSubmitting(true);
    setErrorMessage(null);
    const model = MODEL_OPTIONS[values.provider];
    try {
      const task = await createResearchTask({
        task: values.task,
        modelProvider: values.provider,
        modelId: model.modelId,
        files: nativeFiles,
      });
      workspaceState.registerTask(task);
      setFileList([]);
      setUploadErrorMessage(null);
      form.setFieldValue("task", "");
    } catch (error) {
      setErrorMessage(
        error instanceof ApiClientError ? error.message : "提交失败，请稍后重试",
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: "#0e7490",
          colorInfo: "#0e7490",
          colorSuccess: "#138a66",
          colorText: "#102a43",
          colorTextSecondary: "#52667a",
          borderRadius: 12,
          fontFamily:
            '"Inter", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif',
        },
        components: {
          Button: { controlHeightLG: 50, fontWeight: 650 },
          Input: { controlHeightLG: 48 },
        },
      }}
    >
      <div className="app-shell">
        <header className="topbar">
          <a className="brand" href="#main" aria-label="知研首页">
            <span className="brand-mark" aria-hidden="true">知</span>
            <span>
              <strong>知研</strong>
              <small>DEEP RESEARCH</small>
            </span>
          </a>
          <div className="privacy-note">
            <SafetyCertificateOutlined />
            <span>本地单用户 · 密钥仅保存在后端</span>
          </div>
        </header>

        <main id="main" className="workspace">
          <section className="intro-panel" aria-labelledby="page-title">
            <Tag variant="filled" className="eyebrow">个人深度研究助手</Tag>
            <Title id="page-title" level={1}>
              <span className="title-line title-line-primary">把一个研究问题，</span>
              <span className="title-line title-line-accent">变成有依据的中文报告。</span>
            </Title>
            <Paragraph className="intro-copy">
              Agent 会检索并分析网页资料，也可以结合你的本地文档，通过 FAISS
              找到相关证据，最后生成结构化 Markdown 报告。
            </Paragraph>

            <div className="research-flow" aria-label="研究流程">
              <div className="flow-item">
                <span className="flow-icon"><GlobalOutlined /></span>
                <div><strong>检索网页</strong><small>搜索并抓取公开资料</small></div>
              </div>
              <span className="flow-line" aria-hidden="true" />
              <div className="flow-item">
                <span className="flow-icon"><FileSearchOutlined /></span>
                <div><strong>结合证据</strong><small>分析网页与本地文档</small></div>
              </div>
              <span className="flow-line" aria-hidden="true" />
              <div className="flow-item">
                <span className="flow-icon"><FileTextOutlined /></span>
                <div><strong>生成报告</strong><small>输出结构化研究结论</small></div>
              </div>
            </div>
          </section>

          <section className="research-card" aria-labelledby="form-title">
            <div className="card-heading">
              <div>
                <Text className="step-label">新建研究</Text>
                <Title id="form-title" level={2}>从你的问题开始</Title>
              </div>
              <span className="step-number">01</span>
            </div>

            <Form<ResearchFormValues>
              form={form}
              layout="vertical"
              initialValues={{ provider: "qwen" }}
              requiredMark={false}
              onFinish={(values) => void submitResearch(values)}
              onValuesChange={() => {
                setErrorMessage(null);
              }}
            >
              <Form.Item
                label="研究主题"
                name="task"
                rules={[
                  { required: true, whitespace: true, message: "请输入研究主题" },
                  { max: 4000, message: "研究主题不能超过 4000 个字符" },
                ]}
              >
                <Input.TextArea
                  aria-label="研究主题"
                  rows={4}
                  maxLength={4000}
                  showCount
                  disabled={controlsDisabled}
                  placeholder="例如：调研 RAG 系统中的幻觉问题，比较主流缓解方案并给出实践建议。"
                />
              </Form.Item>

              <Form.Item label="选择研究模型" name="provider">
                <Radio.Group className="model-grid" disabled={controlsDisabled}>
                  {(Object.entries(MODEL_OPTIONS) as [ModelProvider, (typeof MODEL_OPTIONS)[ModelProvider]][]).map(
                    ([provider, option]) => (
                      <Radio.Button key={provider} value={provider} className="model-option">
                        <span className="model-option-content">
                          <span className={`model-mark model-mark-${provider}`}>{option.mark}</span>
                          <span className="model-copy">
                            <strong>{option.title}</strong>
                            <small>{option.description}</small>
                          </span>
                          {selectedProvider === provider && (
                            <CheckCircleFilled className="model-check" aria-hidden="true" />
                          )}
                        </span>
                      </Radio.Button>
                    ),
                  )}
                </Radio.Group>
              </Form.Item>

              <Form.Item
                label={
                  <span>
                    本地资料 <Text type="secondary">（可选）</Text>
                  </span>
                }
                extra="支持 Markdown、TXT、PDF、DOCX；最多 5 个，单个不超过 10 MB，总量不超过 25 MB；每个文件解析后最多 20 万个字符。"
              >
                <Dragger {...uploadProps} className="file-dragger">
                  <p className="ant-upload-drag-icon"><InboxOutlined /></p>
                  <p className="ant-upload-text">点击或拖入研究资料</p>
                  <p className="ant-upload-hint">文件将在开始研究时上传，仅用于本次研究</p>
                </Dragger>
                {uploadErrorMessage && (
                  <Alert
                    className="upload-alert"
                    type="warning"
                    showIcon
                    title="资料添加失败"
                    description={uploadErrorMessage}
                  />
                )}
              </Form.Item>

              {errorMessage && (
                <Alert
                  className="result-alert"
                  type="error"
                  showIcon
                  title="任务创建失败"
                  description={errorMessage}
                  role="alert"
                />
              )}
              <Button
                className="submit-button"
                type="primary"
                htmlType="submit"
                size="large"
                block
                loading={submitting}
                disabled={controlsDisabled}
                iconPlacement="end"
                icon={<ArrowRightOutlined />}
              >
                {submitting
                  ? "正在上传资料并创建任务"
                  : workspaceState.hasActiveTask
                    ? "当前研究正在进行"
                    : "开始深度研究"}
              </Button>
            </Form>

            <div className="card-footer">
              <span>预计耗时取决于主题复杂度和资料数量</span>
              <span>同一时间仅运行一个研究任务</span>
            </div>
          </section>
        </main>

        <section className="task-dashboard" aria-label="研究任务工作区">
          <div className="dashboard-heading">
            <div>
              <Text className="step-label">研究工作区</Text>
              <Title level={2}>跟踪任务，查看最终报告</Title>
            </div>
            <span className="step-number">02</span>
          </div>
          <div className="dashboard-grid">
            <TaskWorkspace
              task={workspaceState.selectedTask}
              report={workspaceState.report}
              loadingReport={workspaceState.loadingReport}
              cancelling={workspaceState.cancelling}
              workspaceError={workspaceState.workspaceError}
              reportError={workspaceState.reportError}
              pollingStopped={workspaceState.pollingStopped}
              onCancel={() => void workspaceState.cancelSelectedTask()}
              onRetry={() => void workspaceState.retrySelectedTask()}
            />
            <RecentTasks
              tasks={workspaceState.tasks}
              selectedTaskId={workspaceState.selectedTask?.id ?? null}
              loading={workspaceState.loadingHistory}
              onSelect={workspaceState.selectTask}
              onRefresh={() => void workspaceState.refreshTasks()}
            />
          </div>
        </section>
      </div>
    </ConfigProvider>
  );
}

export default App;
