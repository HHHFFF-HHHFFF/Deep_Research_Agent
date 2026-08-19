import { ClockCircleOutlined, ReloadOutlined } from "@ant-design/icons";
import { Button, Empty, Skeleton, Tag, Typography } from "antd";

import type { ResearchTask, TaskStatus } from "../api";

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

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

interface RecentTasksProps {
  tasks: ResearchTask[];
  selectedTaskId: string | null;
  loading: boolean;
  onSelect: (task: ResearchTask) => void;
  onRefresh: () => void;
}

export function RecentTasks({ tasks, selectedTaskId, loading, onSelect, onRefresh }: RecentTasksProps) {
  return (
    <aside className="history-panel" aria-labelledby="history-title">
      <div className="panel-heading compact-heading">
        <div>
          <Text className="step-label">最近任务</Text>
          <Title id="history-title" level={3}>研究记录</Title>
        </div>
        <Button
          type="text"
          icon={<ReloadOutlined />}
          onClick={onRefresh}
          loading={loading}
          aria-label="刷新最近任务"
        />
      </div>

      {loading && tasks.length === 0 ? (
        <Skeleton active paragraph={{ rows: 4 }} title={false} />
      ) : tasks.length === 0 ? (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="还没有研究记录" />
      ) : (
        <div className="history-list">
          {tasks.map((task) => (
            <button
              type="button"
              key={task.id}
              className={`history-item${task.id === selectedTaskId ? " history-item-selected" : ""}`}
              onClick={() => onSelect(task)}
              aria-pressed={task.id === selectedTaskId}
            >
              <span className="history-item-top">
                <Tag color={STATUS_COLORS[task.status]}>{STATUS_LABELS[task.status]}</Tag>
                <span className="history-time"><ClockCircleOutlined /> {formatDate(task.created_at)}</span>
              </span>
              <strong>{task.task}</strong>
              <small>{task.model_provider === "qwen" ? "Qwen" : "DeepSeek"}</small>
            </button>
          ))}
        </div>
      )}
    </aside>
  );
}
