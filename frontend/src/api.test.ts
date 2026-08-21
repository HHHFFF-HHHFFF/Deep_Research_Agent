import { beforeEach, describe, expect, it, vi } from "vitest";

import "./test/setup";
import {
  cancelResearchTask,
  createResearchTask,
  getResearchReport,
  getResearchTask,
  listResearchTasks,
  researchReportUrl,
} from "./api";

const taskResponse = {
  id: "task-001",
  task: "比较两种 RAG 方案",
  model_provider: "qwen",
  model_id: "qwen3-max",
  actual_model_name: null,
  status: "waiting",
  stage: "waiting",
  message: "研究任务已进入等待队列",
  error_message: null,
  files: [],
  rag_enabled: false,
  report_available: false,
  created_at: "2026-08-19T00:00:00Z",
  started_at: null,
  finished_at: null,
};

describe("研究接口客户端", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("先上传资料，再使用文件编号创建任务", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: "550e8400-e29b-41d4-a716-446655440000",
            name: "资料.md",
            size: 12,
            created_at: "2026-08-19T00:00:00Z",
          }),
          { status: 201, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(taskResponse), {
          status: 202,
          headers: { "Content-Type": "application/json" },
        }),
      );

    const result = await createResearchTask({
      task: "  比较两种 RAG 方案  ",
      modelProvider: "qwen",
      modelId: "qwen3-max",
      files: [new File(["测试资料"], "资料.md", { type: "text/markdown" })],
    });

    expect(result.id).toBe("task-001");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/files");
    const request = fetchMock.mock.calls[1]?.[1] as RequestInit;
    expect(JSON.parse(request.body as string)).toEqual({
      task: "比较两种 RAG 方案",
      model_provider: "qwen",
      model_id: "qwen3-max",
      file_ids: ["550e8400-e29b-41d4-a716-446655440000"],
    });
  });

  it("把后端安全错误传给界面", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(
        JSON.stringify({ error: { code: "task_busy", message: "已有研究任务正在运行" } }),
        { status: 409, headers: { "Content-Type": "application/json" } },
      ),
    );

    await expect(
      createResearchTask({
        task: "新的研究主题",
        modelProvider: "deepseek",
        modelId: "deepseek-v4-flash",
        files: [],
      }),
    ).rejects.toMatchObject({
      code: "task_busy",
      message: "已有研究任务正在运行",
      status: 409,
    });
  });

  it("网络不可用时返回稳定中文提示", async () => {
    vi.mocked(fetch).mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(
      createResearchTask({
        task: "新的研究主题",
        modelProvider: "qwen",
        modelId: "qwen3-max",
        files: [],
      }),
    ).rejects.toMatchObject({
      code: "service_unavailable",
      message: "无法连接研究服务，请确认后端已经启动",
    });
  });

  it("支持最近任务、详情、取消和报告接口", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ items: [taskResponse] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(taskResponse), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ ...taskResponse, stage: "cancelling" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(new Response("# 研究报告", { status: 200 }));

    expect(await listResearchTasks(5)).toHaveLength(1);
    expect((await getResearchTask("task-001")).id).toBe("task-001");
    expect((await cancelResearchTask("task-001")).stage).toBe("cancelling");
    expect(await getResearchReport("task-001")).toBe("# 研究报告");
    expect(researchReportUrl("task-001")).toBe("/api/tasks/task-001/report");
    expect(fetchMock.mock.calls.map(([input]) => input)).toEqual([
      "/api/tasks?limit=5",
      "/api/tasks/task-001",
      "/api/tasks/task-001/cancel",
      "/api/tasks/task-001/report",
    ]);
  });
});
