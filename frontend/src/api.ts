export type ModelProvider = "qwen" | "deepseek";

export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  created_at: string;
}

export interface ResearchTask {
  id: string;
  task: string;
  model_provider: ModelProvider;
  model_id: string;
  actual_model_name: string | null;
  status:
    | "waiting"
    | "running"
    | "succeeded"
    | "failed"
    | "cancelled"
    | "interrupted";
  stage: string;
  message: string;
  error_message: string | null;
  files: UploadedFile[];
  report_available: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

interface ApiErrorPayload {
  error?: {
    code?: string;
    message?: string;
  };
}

interface CreateTaskInput {
  task: string;
  modelProvider: ModelProvider;
  modelId: string;
  files: File[];
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiClientError extends Error {
  readonly code: string;
  readonly status: number | null;

  constructor(message: string, code = "request_failed", status: number | null = null) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.status = status;
  }
}

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function readError(response: Response): Promise<ApiClientError> {
  let payload: ApiErrorPayload = {};
  try {
    payload = (await response.json()) as ApiErrorPayload;
  } catch {
    // 非 JSON 错误也只向界面提供稳定摘要。
  }
  const message = payload.error?.message || "研究服务暂时无法处理请求";
  const code = payload.error?.code || "request_failed";
  return new ApiClientError(message, code, response.status);
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), init);
  } catch {
    throw new ApiClientError("无法连接研究服务，请确认后端已经启动", "service_unavailable");
  }
  if (!response.ok) {
    throw await readError(response);
  }
  return (await response.json()) as T;
}

export async function uploadResearchFile(file: File): Promise<UploadedFile> {
  const body = new FormData();
  body.append("file", file);
  return requestJson<UploadedFile>("/api/files", {
    method: "POST",
    body,
  });
}

export async function createResearchTask(input: CreateTaskInput): Promise<ResearchTask> {
  const uploadedFiles: UploadedFile[] = [];
  for (const file of input.files) {
    uploadedFiles.push(await uploadResearchFile(file));
  }

  return requestJson<ResearchTask>("/api/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      task: input.task.trim(),
      model_provider: input.modelProvider,
      model_id: input.modelId,
      file_ids: uploadedFiles.map((file) => file.id),
    }),
  });
}
