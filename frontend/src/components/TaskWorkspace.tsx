import {
  CloseCircleOutlined,
  DownloadOutlined,
  FileTextOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import { Alert, Button, Empty, Skeleton, Space, Tag, Typography } from "antd";
import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

import { researchReportUrl, type ResearchTask, type TaskStatus } from "../api";
import { isTaskActive } from "../useResearchWorkspace";

const { Text, Title } = Typography;

const STATUS_LABELS: Record<TaskStatus, string> = {
  waiting: "等待中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
};

const STATUS_COLORS: Record<TaskStatus, string> = {
  waiting: "gold",
  running: "cyan",
  succeeded: "green",
  failed: "red",
  cancelled: "default",
  interrupted: "orange",
};

const STAGE_LABELS: Record<string, string> = {
  waiting: "等待研究资源",
  initializing: "正在初始化研究组件",
  researching: "正在检索、分析并撰写报告",
  cancelling: "正在安全取消任务",
  completed: "研究报告已经生成",
  failed: "研究任务执行失败",
  cancelled: "研究任务已取消",
  interrupted: "研究任务因服务重启而中断",
};

function formatDuration(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes} 分 ${seconds} 秒` : `${seconds} 秒`;
}

function useTaskDuration(task: ResearchTask | null): string {
  const [now, setNow] = useState(Date.now());
  const active = isTaskActive(task);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active]);

  return useMemo(() => {
    if (!task) return "0 秒";
    const start = new Date(task.started_at ?? task.created_at).getTime();
    const end = task.finished_at ? new Date(task.finished_at).getTime() : now;
    return formatDuration(end - start);
  }, [now, task]);
}

interface TaskWorkspaceProps {
  task: ResearchTask | null;
  report: string | null;
  loadingReport: boolean;
  cancelling: boolean;
  workspaceError: string | null;
  reportError: string | null;
  pollingStopped: boolean;
  onCancel: () => void;
  onRetry: () => void;
}

export function TaskWorkspace({
  task,
  report,
  loadingReport,
  cancelling,
  workspaceError,
  reportError,
  pollingStopped,
  onCancel,
  onRetry,
}: TaskWorkspaceProps) {
  const duration = useTaskDuration(task);

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
            <div className={`stage-message${isTaskActive(task) ? " stage-message-active" : ""}`}>
              <span className="stage-dot" aria-hidden="true" />
              <span>{task.message}</span>
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
              {isTaskActive(task) && (
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
