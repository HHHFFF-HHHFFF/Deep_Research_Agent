import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import "./test/setup";
import App from "./App";

function successfulTaskResponse() {
  return new Response(
    JSON.stringify({
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
    }),
    { status: 202, headers: { "Content-Type": "application/json" } },
  );
}

describe("研究输入页面", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("拒绝空主题且不会调用后端", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.getByText("适合中文研究与工具调用")).toBeVisible();
    expect(screen.getByText("侧重推理与内容分析")).toBeVisible();
    await user.click(screen.getByRole("button", { name: /开始深度研究/ }));

    expect(await screen.findByText("请输入研究主题")).toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("可以选择模型、添加资料并创建研究任务", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    fetchMock
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: "550e8400-e29b-41d4-a716-446655440000",
            name: "资料.txt",
            size: 12,
            created_at: "2026-08-19T00:00:00Z",
          }),
          { status: 201, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(successfulTaskResponse());
    render(<App />);

    await user.type(screen.getByLabelText("研究主题"), "研究主题");
    await user.click(screen.getByText("DeepSeek"));
    const fileInput = document.querySelector<HTMLInputElement>('input[type="file"]');
    expect(fileInput).not.toBeNull();
    fireEvent.change(fileInput!, {
      target: { files: [new File(["本地资料"], "资料.txt", { type: "text/plain" })] },
    });
    await user.click(screen.getByRole("button", { name: /开始深度研究/ }));

    expect(await screen.findByText("研究任务已创建")).toBeInTheDocument();
    expect(screen.getByText(/task-frontend-001/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const createRequest = fetchMock.mock.calls[1]?.[1] as RequestInit;
    expect(JSON.parse(createRequest.body as string)).toMatchObject({
      task: "研究主题",
      model_provider: "deepseek",
      model_id: "deepseek-v4-flash",
    });
  });

  it("服务不可用时恢复按钮并显示清晰提示", async () => {
    const user = userEvent.setup();
    vi.mocked(fetch).mockRejectedValue(new TypeError("Failed to fetch"));
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
