import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import "./test/setup";
import App from "./App";
import type { ResearchTask } from "./api";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function taskResponse(overrides: Partial<ResearchTask> = {}): ResearchTask {
  return {
    id: "task-frontend-001",
    task: "研究主题",
    model_provider: "deepseek",
    model_id: "deepseek-v4-flash",
    actual_model_name: null,
    status: "waiting",
    stage: "waiting",
    message: "研究任务已创建，正在等待执行",
    error_message: null,
    files: [],
    report_available: false,
    created_at: "2026-08-19T00:00:00Z",
    started_at: null,
    finished_at: null,
    ...overrides,
  };
}

function useEmptyHistory(): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    if (String(input).startsWith("/api/tasks?")) return jsonResponse({ items: [] });
    throw new Error(`未处理的测试请求：${String(input)}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("研究输入与任务工作区", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("拒绝空主题且不会创建任务", async () => {
    const user = userEvent.setup();
    const fetchMock = useEmptyHistory();
    render(<App />);

    expect(screen.getByText("适合中文研究与工具调用")).toBeVisible();
    expect(screen.getByText("侧重推理与内容分析")).toBeVisible();
    await user.click(screen.getByRole("button", { name: /开始深度研究/ }));

    expect(await screen.findByText("请输入研究主题")).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(([input, init]) => (
        String(input) === "/api/tasks" && (init as RequestInit | undefined)?.method === "POST"
      )),
    ).toBe(false);
  });

  it("可以选择模型、添加资料并创建研究任务", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.startsWith("/api/tasks?")) return jsonResponse({ items: [] });
      if (path === "/api/files") {
        return jsonResponse({
          id: "550e8400-e29b-41d4-a716-446655440000",
          name: "资料.txt",
          size: 12,
          created_at: "2026-08-19T00:00:00Z",
        }, 201);
      }
      if (path === "/api/tasks" && init?.method === "POST") {
        return jsonResponse(taskResponse(), 202);
      }
      throw new Error(`未处理的测试请求：${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    await user.type(screen.getByLabelText("研究主题"), "研究主题");
    await user.click(screen.getByText("DeepSeek"));
    const fileInput = document.querySelector<HTMLInputElement>('input[type="file"]');
    expect(fileInput).not.toBeNull();
    fireEvent.change(fileInput!, {
      target: { files: [new File(["本地资料"], "资料.txt", { type: "text/plain" })] },
    });
    await user.click(screen.getByRole("button", { name: /开始深度研究/ }));

    expect(await screen.findByText("研究任务已创建，正在等待执行")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /当前研究正在进行/ })).toBeDisabled();
    const createCall = fetchMock.mock.calls.find(([input, init]) => (
      String(input) === "/api/tasks" && init?.method === "POST"
    ));
    expect(createCall).toBeDefined();
    expect(JSON.parse(createCall?.[1]?.body as string)).toMatchObject({
      task: "研究主题",
      model_provider: "deepseek",
      model_id: "deepseek-v4-flash",
    });
  });

  it("恢复已完成任务并安全展示和下载报告", async () => {
    const completedTask = taskResponse({
      id: "completed-task",
      status: "succeeded",
      stage: "completed",
      message: "研究报告已经生成",
      report_available: true,
      started_at: "2026-08-19T00:00:01Z",
      finished_at: "2026-08-19T00:00:06Z",
    });
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.startsWith("/api/tasks?")) return jsonResponse({ items: [completedTask] });
      if (path === "/api/tasks/completed-task/report") {
        return new Response("# 研究结论\n\n报告正文\n\n<script>恶意脚本</script>", { status: 200 });
      }
      throw new Error(`未处理的测试请求：${path}`);
    }));

    render(<App />);

    expect(await screen.findByRole(
      "heading",
      { name: "研究结论" },
      { timeout: 5_000 },
    )).toBeInTheDocument();
    expect(screen.getByText("报告正文")).toBeInTheDocument();
    expect(screen.queryByText("恶意脚本")).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: /下载 Markdown/ })).toHaveAttribute(
      "href",
      "/api/tasks/completed-task/report",
    );
  });

  it("可以对运行中的任务发出协作式取消", async () => {
    const user = userEvent.setup();
    const runningTask = taskResponse({
      id: "running-task",
      status: "running",
      stage: "researching",
      message: "正在执行深度研究",
      started_at: "2026-08-19T00:00:01Z",
    });
    const cancellingTask = taskResponse({
      ...runningTask,
      stage: "cancelling",
      message: "已发送取消请求，正在安全停止",
    });
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.startsWith("/api/tasks?")) return jsonResponse({ items: [runningTask] });
      if (path === "/api/tasks/running-task/cancel" && init?.method === "POST") {
        return jsonResponse(cancellingTask);
      }
      throw new Error(`未处理的测试请求：${path}`);
    }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: /取消研究/ }));

    expect(await screen.findByText("正在安全取消任务")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /正在取消/ })).toBeDisabled();
  });

  it("服务不可用时恢复按钮并显示清晰提示", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).startsWith("/api/tasks?")) return jsonResponse({ items: [] });
      throw new TypeError("Failed to fetch");
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    await user.type(screen.getByLabelText("研究主题"), "研究一个新方向");
    await user.click(screen.getByRole("button", { name: /开始深度研究/ }));

    expect(await screen.findByText("任务创建失败")).toBeInTheDocument();
    expect(screen.getByText("无法连接研究服务，请确认后端已经启动")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /开始深度研究/ })).toBeEnabled();
    });
  });
});
