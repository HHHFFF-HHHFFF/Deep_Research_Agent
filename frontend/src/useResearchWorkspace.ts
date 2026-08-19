import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiClientError,
  cancelResearchTask,
  getResearchReport,
  getResearchTask,
  listResearchTasks,
  type ResearchTask,
} from "./api";

const LAST_TASK_STORAGE_KEY = "deep-research-agent:last-task-id";
const DEFAULT_POLL_INTERVAL_MS = 1500;
const DEFAULT_MAX_POLL_FAILURES = 3;
const DEFAULT_HISTORY_LIMIT = 8;

export function isTaskActive(task: ResearchTask | null): boolean {
  return task?.status === "waiting" || task?.status === "running";
}

function safeMessage(error: unknown, fallback: string): string {
  return error instanceof ApiClientError ? error.message : fallback;
}

function readLastTaskId(): string | null {
  try {
    return window.localStorage.getItem(LAST_TASK_STORAGE_KEY);
  } catch {
    return null;
  }
}

function saveLastTaskId(taskId: string): void {
  try {
    window.localStorage.setItem(LAST_TASK_STORAGE_KEY, taskId);
  } catch {
    // 浏览器禁用存储时仍然可以使用当前页面，不影响研究任务。
  }
}

interface ResearchWorkspaceOptions {
  pollIntervalMs?: number;
  maxPollFailures?: number;
  historyLimit?: number;
}

export function useResearchWorkspace(options: ResearchWorkspaceOptions = {}) {
  const pollIntervalMs = options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const maxPollFailures = options.maxPollFailures ?? DEFAULT_MAX_POLL_FAILURES;
  const historyLimit = options.historyLimit ?? DEFAULT_HISTORY_LIMIT;
  const [tasks, setTasks] = useState<ResearchTask[]>([]);
  const [selectedTask, setSelectedTask] = useState<ResearchTask | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [loadingReport, setLoadingReport] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);
  const [pollingStopped, setPollingStopped] = useState(false);
  const mountedRef = useRef(true);

  useEffect(() => () => {
    mountedRef.current = false;
  }, []);

  const applyTask = useCallback((task: ResearchTask) => {
    if (!mountedRef.current) return;
    setSelectedTask(task);
    setTasks((current) => [task, ...current.filter((item) => item.id !== task.id)].slice(0, historyLimit));
    saveLastTaskId(task.id);
  }, [historyLimit]);

  const selectTask = useCallback((task: ResearchTask) => {
    setSelectedTask(task);
    setWorkspaceError(null);
    setReportError(null);
    setPollingStopped(false);
    saveLastTaskId(task.id);
  }, []);

  const refreshTasks = useCallback(async () => {
    setLoadingHistory(true);
    try {
      const recentTasks = await listResearchTasks(historyLimit);
      if (!mountedRef.current) return;
      setTasks(recentTasks);
      setWorkspaceError(null);

      const savedTaskId = readLastTaskId();
      const activeTask = recentTasks.find(isTaskActive);
      let preferredTask: ResearchTask | null = activeTask
        ?? recentTasks.find((task) => task.id === savedTaskId)
        ?? null;

      if (!preferredTask && savedTaskId) {
        try {
          preferredTask = await getResearchTask(savedTaskId);
        } catch {
          preferredTask = null;
        }
      }
      preferredTask ??= recentTasks.at(0) ?? null;
      if (mountedRef.current && preferredTask) selectTask(preferredTask);
    } catch (error) {
      if (mountedRef.current) {
        setWorkspaceError(safeMessage(error, "最近任务加载失败"));
      }
    } finally {
      if (mountedRef.current) setLoadingHistory(false);
    }
  }, [historyLimit, selectTask]);

  useEffect(() => {
    void refreshTasks();
  }, [refreshTasks]);

  useEffect(() => {
    if (!selectedTask || !isTaskActive(selectedTask) || pollingStopped) return;

    let disposed = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    let failureCount = 0;

    const schedule = () => {
      timeoutId = setTimeout(() => void poll(), pollIntervalMs);
    };
    const poll = async () => {
      controller = new AbortController();
      try {
        const task = await getResearchTask(selectedTask.id, controller.signal);
        if (disposed) return;
        failureCount = 0;
        setWorkspaceError(null);
        applyTask(task);
        if (isTaskActive(task)) schedule();
      } catch (error) {
        if (disposed || (error instanceof Error && error.name === "AbortError")) return;
        failureCount += 1;
        if (failureCount >= maxPollFailures) {
          setWorkspaceError("研究服务连续连接失败，状态更新已暂停");
          setPollingStopped(true);
          return;
        }
        schedule();
      }
    };

    schedule();
    return () => {
      disposed = true;
      if (timeoutId) clearTimeout(timeoutId);
      controller?.abort();
    };
  }, [applyTask, maxPollFailures, pollIntervalMs, pollingStopped, selectedTask]);

  useEffect(() => {
    setReport(null);
    setReportError(null);
    if (!selectedTask?.report_available) {
      setLoadingReport(false);
      return;
    }

    const controller = new AbortController();
    setLoadingReport(true);
    void getResearchReport(selectedTask.id, controller.signal)
      .then((content) => {
        if (mountedRef.current) setReport(content);
      })
      .catch((error) => {
        if (error instanceof Error && error.name === "AbortError") return;
        if (mountedRef.current) {
          setReportError(safeMessage(error, "研究报告加载失败"));
        }
      })
      .finally(() => {
        if (mountedRef.current && !controller.signal.aborted) setLoadingReport(false);
      });
    return () => controller.abort();
  }, [selectedTask?.id, selectedTask?.report_available]);

  const registerTask = useCallback((task: ResearchTask) => {
    setPollingStopped(false);
    setWorkspaceError(null);
    applyTask(task);
  }, [applyTask]);

  const cancelSelectedTask = useCallback(async () => {
    if (!selectedTask || !isTaskActive(selectedTask) || cancelling) return;
    setCancelling(true);
    setWorkspaceError(null);
    try {
      applyTask(await cancelResearchTask(selectedTask.id));
    } catch (error) {
      setWorkspaceError(safeMessage(error, "取消研究任务失败"));
    } finally {
      if (mountedRef.current) setCancelling(false);
    }
  }, [applyTask, cancelling, selectedTask]);

  const retrySelectedTask = useCallback(async () => {
    setWorkspaceError(null);
    setPollingStopped(false);
    if (!selectedTask) {
      await refreshTasks();
      return;
    }
    try {
      applyTask(await getResearchTask(selectedTask.id));
    } catch (error) {
      setWorkspaceError(safeMessage(error, "研究任务状态加载失败"));
    }
  }, [applyTask, refreshTasks, selectedTask]);

  const hasActiveTask = useMemo(
    () => isTaskActive(selectedTask) || tasks.some(isTaskActive),
    [selectedTask, tasks],
  );

  return {
    tasks,
    selectedTask,
    report,
    loadingHistory,
    loadingReport,
    cancelling,
    workspaceError,
    reportError,
    pollingStopped,
    hasActiveTask,
    registerTask,
    selectTask,
    refreshTasks,
    cancelSelectedTask,
    retrySelectedTask,
  };
}
