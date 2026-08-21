import { useState } from "react";
import { ClockCircleOutlined, DeleteOutlined, ReloadOutlined } from "@ant-design/icons";
import { Button, Empty, Modal, Skeleton, Tag, Tooltip, Typography } from "antd";

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
  deletingTaskId: string | null;
  onSelect: (task: ResearchTask) => void;
  onDelete: (task: ResearchTask) => Promise<void>;
  onRefresh: () => void;
}

function isTaskActive(task: ResearchTask): boolean {
  return task.status === "waiting" || task.status === "running";
}

export function RecentTasks({
  tasks,
  selectedTaskId,
  loading,
  deletingTaskId,
  onSelect,
  onDelete,
  onRefresh,
}: RecentTasksProps) {
  const [pendingDelete, setPendingDelete] = useState<ResearchTask | null>(null);

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
            <div
              key={task.id}
              className={`history-item${task.id === selectedTaskId ? " history-item-selected" : ""}`}
            >
              <button
                type="button"
                className="history-item-main"
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
              {isTaskActive(task) ? (
                <Tooltip title="运行中的任务不能删除">
                  <Button
                    className="history-delete-button"
                    type="text"
                    size="small"
                    disabled
                    icon={<DeleteOutlined />}
                    aria-label={`暂时不能删除：${task.task}`}
                  />
                </Tooltip>
              ) : (
                <Button
                  className="history-delete-button"
                  type="text"
                  danger
                  size="small"
                  loading={deletingTaskId === task.id}
                  disabled={deletingTaskId !== null}
                  icon={<DeleteOutlined />}
                  aria-label={`删除研究记录：${task.task}`}
                  onClick={() => setPendingDelete(task)}
                />
              )}
            </div>
          ))}
        </div>
      )}
      <Modal
        title="删除这条研究记录？"
        open={pendingDelete !== null}
        okText="删除"
        cancelText="取消"
        okButtonProps={{ danger: true }}
        confirmLoading={pendingDelete?.id === deletingTaskId}
        onCancel={() => setPendingDelete(null)}
        onOk={async () => {
          if (!pendingDelete) return;
          await onDelete(pendingDelete);
          setPendingDelete(null);
        }}
      >
        <p>对应报告和未被其他任务使用的资料也会删除，且无法恢复。</p>
      </Modal>
    </aside>
  );
}
