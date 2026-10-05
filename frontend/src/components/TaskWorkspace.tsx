import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  DownloadOutlined,
  FileTextOutlined,
  FileSearchOutlined,
  LinkOutlined,
  LoadingOutlined,
  ReloadOutlined,
  ToolOutlined,
  UnorderedListOutlined,
} from "@ant-design/icons";
import { Alert, Button, Empty, Input, Skeleton, Tag, Typography } from "antd";
import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

import {
  researchReportUrl,
  type ResearchTask,
  type TaskActivityStatus,
  type TaskStatus,
} from "../api";

const { Text, Title } = Typography;

const STATUS_LABELS: Record<TaskStatus, string> = {
  awaiting_confirmation: "待确认",
  waiting: "等待中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
};

const STATUS_COLORS: Record<TaskStatus, string> = {
  awaiting_confirmation: "blue",
  waiting: "gold",
  running: "cyan",
  succeeded: "green",
  failed: "red",
  cancelled: "default",
  interrupted: "orange",
};

const STAGE_LABELS: Record<string, string> = {
  awaiting_confirmation: "等待确认研究计划",
  waiting: "等待研究资源",
  initializing: "正在初始化研究组件",
  researching: "正在检索、分析并撰写报告",
  cancelling: "正在安全取消任务",
  completed: "研究报告已经生成",
  failed: "研究任务执行失败",
  cancelled: "研究任务已取消",
  interrupted: "研究任务因服务重启而中断",
};

const ACTIVITY_STATUS_LABELS: Record<TaskActivityStatus, string> = {
  running: "进行中",
  succeeded: "已完成",
  failed: "未完成",
  cancelled: "已停止",
};

function formatDuration(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes} 分 ${seconds} 秒` : `${seconds} 秒`;
}

function useTaskDuration(task: ResearchTask | null): string {
  const [now, setNow] = useState(Date.now());
  const active = task?.status === "waiting" || task?.status === "running";

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active]);

  return useMemo(() => {
    if (!task) return "0 秒";
    if (task.status === "awaiting_confirmation") return "尚未开始";
    const start = new Date(task.started_at ?? task.created_at).getTime();
    const end = task.finished_at ? new Date(task.finished_at).getTime() : now;
    return formatDuration(end - start);
  }, [now, task]);
}

function formatActivityDuration(milliseconds: number | null): string | null {
  if (milliseconds === null) return null;
  if (milliseconds < 1000) return `${milliseconds} 毫秒`;
  return formatDuration(milliseconds);
}

function formatEvidenceScore(score: number | null): string | null {
  if (score === null) return null;
  return score.toFixed(3);
}

interface TaskWorkspaceProps {
  task: ResearchTask | null;
  report: string | null;
  loadingReport: boolean;
  cancelling: boolean;
  confirmingPlan: boolean;
  workspaceError: string | null;
  reportError: string | null;
  pollingStopped: boolean;
  onCancel: () => void;
  onConfirmPlan: (steps: string[]) => Promise<void>;
  onRetry: () => void;
}

export function TaskWorkspace({
  task,
  report,
  loadingReport,
  cancelling,
  confirmingPlan,
  workspaceError,
  reportError,
  pollingStopped,
  onCancel,
  onConfirmPlan,
  onRetry,
}: TaskWorkspaceProps) {
  const duration = useTaskDuration(task);
  const [planDraft, setPlanDraft] = useState("");
  const [planError, setPlanError] = useState<string | null>(null);

  useEffect(() => {
    setPlanDraft(task?.research_plan.join("\n") ?? "");
    setPlanError(null);
  }, [task?.id, task?.research_plan]);

  const confirmPlan = async () => {
    const steps = planDraft.split("\n").map((step) => step.trim()).filter(Boolean);
    if (steps.length < 2 || steps.length > 8) {
      setPlanError("研究计划需要包含 2～8 个非空步骤，每行一个步骤");
      return;
    }
    if (steps.some((step) => step.length < 3 || step.length > 300)) {
      setPlanError("每个步骤需要包含 3～300 个字符");
      return;
    }
    if (new Set(steps).size !== steps.length) {
      setPlanError("研究计划不能包含重复步骤");
      return;
    }
    setPlanError(null);
    await onConfirmPlan(steps);
  };

  return (
    <section className="task-panel" aria-labelledby="task-panel-title">
      <div className="panel-heading">
        <div>
          <Text className="step-label">任务状态</Text>
          <Title id="task-panel-title" level={3}>研究进度与报告</Title>
        </div>
        {task && <Tag color={STATUS_COLORS[task.status]}>{STATUS_LABELS[task.status]}</Tag>}
      </div>

      {!task ? (
        <div>
          {workspaceError && (
            <Alert
              type="warning"
              showIcon
              title="研究服务暂时不可用"
              description={workspaceError}
              action={<Button size="small" icon={<ReloadOutlined />} onClick={onRetry}>重新连接</Button>}
            />
          )}
          <Empty
            className="task-empty"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="提交研究主题后，可在这里查看进度和报告"
          />
        </div>
      ) : (
        <>
          <div className="task-summary">
            <Title level={4}>{task.task}</Title>
            <div className="task-meta-grid">
              <span><small>当前阶段</small><strong>{STAGE_LABELS[task.stage] ?? task.message}</strong></span>
              <span><small>运行时间</small><strong>{duration}</strong></span>
              <span><small>研究模型</small><strong>{task.actual_model_name ?? task.model_id}</strong></span>
              <span><small>本地 RAG</small><strong>{task.rag_enabled ? "已启用" : "未启用"}</strong></span>
            </div>
            {task.files.length > 0 && (
              <div className="task-files">
                <small>已使用资料</small>
                <div>
                  {task.files.map((file) => <Tag key={file.id}>{file.name}</Tag>)}
                </div>
              </div>
            )}
            <div className={`stage-message${
              task.status === "waiting" || task.status === "running"
                ? " stage-message-active"
                : ""
            }`}>
              <span className="stage-dot" aria-hidden="true" />
              <span>{task.message}</span>
            </div>

            {task.status === "awaiting_confirmation" && (
              <div className="plan-confirmation">
                <div className="plan-confirmation-heading">
                  <span className="plan-confirmation-icon" aria-hidden="true">
                    <UnorderedListOutlined />
                  </span>
                  <div>
                    <strong>执行前确认研究计划</strong>
                    <small>每行代表一个步骤，可直接修改、增加或删除</small>
                  </div>
                </div>
                <Input.TextArea
                  aria-label="研究计划"
                  value={planDraft}
                  onChange={(event) => {
                    setPlanDraft(event.target.value);
                    setPlanError(null);
                  }}
                  autoSize={{ minRows: 5, maxRows: 10 }}
                  maxLength={2400}
                  disabled={confirmingPlan || cancelling}
                />
                {planError && <Text type="danger">{planError}</Text>}
                <div className="plan-confirmation-actions">
                  <Button
                    type="primary"
                    loading={confirmingPlan}
                    disabled={confirmingPlan || cancelling}
                    onClick={() => void confirmPlan()}
                  >
                    确认并开始研究
                  </Button>
                  <Button
                    danger
                    loading={cancelling}
                    disabled={confirmingPlan || cancelling}
                    onClick={onCancel}
                  >
                    取消任务
                  </Button>
                </div>
              </div>
            )}

            <div className="activity-section" aria-live="polite">
              <div className="activity-heading">
                <div>
                  <strong>Agent 执行过程</strong>
                  <small>展示可审阅的行动摘要，不包含模型隐藏推理</small>
                </div>
                <Tag>{task.activities.length} 条记录</Tag>
              </div>
              {task.activities.length > 0 ? (
                <ol className="activity-list">
                  {task.activities.map((activity) => {
                    const durationLabel = formatActivityDuration(activity.duration_ms);
                    return (
                      <li
                        key={activity.id}
                        className={`activity-item activity-item-${activity.status}`}
                      >
                        <span className="activity-icon" aria-hidden="true">
                          {activity.status === "running" ? (
                            <LoadingOutlined spin />
                          ) : activity.status === "succeeded" ? (
                            <CheckCircleOutlined />
                          ) : (
                            <CloseCircleOutlined />
                          )}
                        </span>
                        <div className="activity-content">
                          <div className="activity-title-row">
                            <strong>{activity.title}</strong>
                            <span>{ACTIVITY_STATUS_LABELS[activity.status]}</span>
                          </div>
                          {activity.detail && <small>{activity.detail}</small>}
                          <div className="activity-meta">
                            {activity.step_number !== null && (
                              <span>第 {activity.step_number} 轮</span>
                            )}
                            {activity.tool_name && (
                              <span><ToolOutlined /> {activity.tool_name}</span>
                            )}
                            {durationLabel && <span>耗时 {durationLabel}</span>}
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ol>
              ) : (
                <Text className="activity-empty" type="secondary">
                  任务开始后将在这里显示规划、资料检索和工具执行状态
                </Text>
              )}
            </div>

            <div className="evidence-section">
              <div className="evidence-heading">
                <div>
                  <strong>研究证据与引用核验</strong>
                  <small>网页来源可直接打开，本地引用可跳转到对应检索片段</small>
                </div>
                {task.citation_validation ? (
                  <Tag color={task.citation_validation.passed ? "green" : "orange"}>
                    {task.citation_validation.passed ? "引用检查通过" : "引用需要复核"}
                  </Tag>
                ) : (
                  <Tag>等待报告核验</Tag>
                )}
              </div>

              {task.citation_validation && (
                <div className="citation-summary">
                  <span>
                    有效引用
                    <strong>
                      {task.citation_validation.valid_citations}/
                      {task.citation_validation.total_citations}
                    </strong>
                  </span>
                  <span>
                    证据覆盖
                    <strong>
                      {Math.round(task.citation_validation.coverage_rate * 100)}%
                    </strong>
                  </span>
                  <span>
                    已引用证据
                    <strong>
                      {task.citation_validation.cited_evidence}/
                      {task.citation_validation.total_evidence}
                    </strong>
                  </span>
                </div>
              )}

              {task.citation_validation && task.citation_validation.issues.length > 0 && (
                <Alert
                  className="citation-alert"
                  type="warning"
                  showIcon
                  title="引用核验提示"
                  description={(
                    <ul>
                      {task.citation_validation.issues.map((issue, index) => (
                        <li key={`${issue.label}-${index}`}>
                          {issue.label}：{issue.reason}
                        </li>
                      ))}
                    </ul>
                  )}
                />
              )}

              {task.evidence.length > 0 ? (
                <div className="evidence-list">
                  {task.evidence.map((evidence) => {
                    const score = formatEvidenceScore(evidence.relevance_score);
                    const cited = evidence.citation_labels.length > 0;
                    return (
                      <article
                        id={`evidence-${evidence.id}`}
                        key={evidence.id}
                        className="evidence-card"
                      >
                        <div className="evidence-card-heading">
                          <span className="evidence-source-type">
                            {evidence.source_type === "web" ? (
                              <LinkOutlined />
                            ) : (
                              <FileSearchOutlined />
                            )}
                            {evidence.source_type === "web" ? "网页来源" : "本地片段"}
                          </span>
                          <Tag color={cited ? "green" : "default"}>
                            {cited ? "报告已引用" : "暂未引用"}
                          </Tag>
                        </div>
                        {evidence.url ? (
                          <a
                            className="evidence-title"
                            href={evidence.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {evidence.title}
                          </a>
                        ) : (
                          <strong className="evidence-title">{evidence.title}</strong>
                        )}
                        {evidence.excerpt && (
                          <p className="evidence-excerpt">{evidence.excerpt}</p>
                        )}
                        <div className="evidence-meta">
                          {evidence.file_name && <span>{evidence.file_name}</span>}
                          {evidence.chunk_index !== null && (
                            <span>片段 {evidence.chunk_index}</span>
                          )}
                          {score && <span>相关度 {score}</span>}
                          {evidence.citation_labels.map((label) => (
                            <span key={label}>引用 {label}</span>
                          ))}
                        </div>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <Text className="evidence-empty" type="secondary">
                  研究过程中采集到的网页来源和本地检索片段会显示在这里
                </Text>
              )}
            </div>

            {task.error_message && (
              <Alert type="error" showIcon title="研究任务未完成" description={task.error_message} />
            )}
            {workspaceError && (
              <Alert
                type="warning"
                showIcon
                title="任务状态暂时不可用"
                description={workspaceError}
                action={pollingStopped ? (
                  <Button size="small" icon={<ReloadOutlined />} onClick={onRetry}>重新连接</Button>
                ) : undefined}
              />
            )}

            <div className="task-actions">
              {(task.status === "waiting" || task.status === "running") && (
                <Button
                  danger
                  icon={<CloseCircleOutlined />}
                  loading={cancelling}
                  disabled={cancelling || task.stage === "cancelling"}
                  onClick={onCancel}
                >
                  {task.stage === "cancelling" ? "正在取消" : "取消研究"}
                </Button>
              )}
              {task.report_available && (
                <a className="report-download" href={researchReportUrl(task.id)} download>
                  <DownloadOutlined /> 下载 Markdown
                </a>
              )}
            </div>
          </div>

          {task.report_available && (
            <div className="report-section">
              <div className="report-heading"><FileTextOutlined /><strong>研究报告</strong></div>
              {loadingReport ? (
                <Skeleton active paragraph={{ rows: 8 }} />
              ) : reportError ? (
                <Alert type="error" showIcon title="报告加载失败" description={reportError} />
              ) : report ? (
                <article className="markdown-report">
                  <ReactMarkdown
                    skipHtml
                    components={{
                      a: ({ node: _, ...props }) => (
                        <a {...props} target="_blank" rel="noopener noreferrer" />
                      ),
                    }}
                  >
                    {report}
                  </ReactMarkdown>
                </article>
              ) : (
                <Empty description="报告内容为空" />
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
