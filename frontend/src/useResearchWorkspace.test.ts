import { renderHook, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import "./test/setup";
import type { ResearchTask } from "./api";
import { useResearchWorkspace } from "./useResearchWorkspace";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function taskResponse(overrides: Partial<ResearchTask> = {}): ResearchTask {
  return {
    id: "polling-task",
    task: "轮询研究任务",
    model_provider: "qwen",
    model_id: "qwen3-max",
    actual_model_name: null,
    status: "running",
    stage: "researching",
    message: "正在执行研究",
    error_message: null,
    files: [],
    rag_enabled: false,
    report_available: false,
    created_at: "2026-08-19T00:00:00Z",
    started_at: "2026-08-19T00:00:01Z",
    finished_at: null,
    ...overrides,
  };
}

describe("研究工作区状态管理", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("轮询到终态后停止并加载报告", async () => {
    const runningTask = taskResponse();
    const completedTask = taskResponse({
      status: "succeeded",
      stage: "completed",
      message: "研究完成",
      report_available: true,
      finished_at: "2026-08-19T00:00:05Z",
    });
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.startsWith("/api/tasks?")) return jsonResponse({ items: [runningTask] });
      if (path === "/api/tasks/polling-task") return jsonResponse(completedTask);
      if (path === "/api/tasks/polling-task/report") {
        return new Response("# 已完成报告", { status: 200 });
      }
      throw new Error(`未处理的测试请求：${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const { result, unmount } = renderHook(() => useResearchWorkspace({ pollIntervalMs: 10 }));

    await waitFor(() => expect(result.current.selectedTask?.status).toBe("succeeded"));
    await waitFor(() => expect(result.current.report).toBe("# 已完成报告"));
    const detailCalls = fetchMock.mock.calls.filter(
      ([input]) => String(input) === "/api/tasks/polling-task",
    );
    expect(detailCalls).toHaveLength(1);
    unmount();
  });

  it("React 严格模式重新挂载后仍能结束最近任务加载", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).startsWith("/api/tasks?")) return jsonResponse({ items: [] });
      throw new Error(`未处理的测试请求：${String(input)}`);
    }));

    const { result } = renderHook(() => useResearchWorkspace(), {
      wrapper: StrictMode,
    });

    await waitFor(() => expect(result.current.loadingHistory).toBe(false));
    expect(result.current.workspaceError).toBeNull();
  });

  it("连续连接失败三次后暂停轮询", async () => {
    const runningTask = taskResponse();
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).startsWith("/api/tasks?")) {
        return jsonResponse({ items: [runningTask] });
      }
      throw new TypeError("Failed to fetch");
    }));

    const { result } = renderHook(() => useResearchWorkspace({
      pollIntervalMs: 5,
      maxPollFailures: 3,
    }));

    await waitFor(() => expect(result.current.pollingStopped).toBe(true));
    expect(result.current.workspaceError).toBe("研究服务连续连接失败，状态更新已暂停");
  });

  it("最近列表中不存在时仍按本地任务编号恢复", async () => {
    const savedTask = taskResponse({
      id: "saved-task",
      status: "succeeded",
      stage: "completed",
      report_available: true,
      finished_at: "2026-08-19T00:00:05Z",
    });
    window.localStorage.setItem("deep-research-agent:last-task-id", "saved-task");
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.startsWith("/api/tasks?")) return jsonResponse({ items: [] });
      if (path === "/api/tasks/saved-task") return jsonResponse(savedTask);
      if (path === "/api/tasks/saved-task/report") {
        return new Response("# 恢复的报告", { status: 200 });
      }
      throw new Error(`未处理的测试请求：${path}`);
    }));

    const { result } = renderHook(() => useResearchWorkspace({ pollIntervalMs: 10 }));

    await waitFor(() => expect(result.current.selectedTask?.id).toBe("saved-task"));
    await waitFor(() => expect(result.current.report).toBe("# 恢复的报告"));
  });
});
