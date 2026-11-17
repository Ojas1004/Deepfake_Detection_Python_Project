// Types and calls for the Python API (src/api.py).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export type BenchmarkSummary = {
  openfaketiny?: { fakes_caught: number; reals_kept: number; roc_auc: number };
  frontier?: { fakes_caught: number; reals_kept: number; roc_auc: number; detected_at_1pct_fp: number };
};

export type Model = {
  id: string;
  label: string;
  short: string;
  kind: "ours" | "external";
  architecture: string;
  input: string;
  sure_fake: number;
  calibrated: boolean;
  heatmap: string;
  licence?: string;
  test: {
    accuracy: number;
    roc_auc: number;
    precision: number;
    recall: number;
    n_images: number;
    unseen: { name: string; detection_rate: number }[];
  } | null;
  benchmarks: BenchmarkSummary;
  note: string | null;
};

export type Region = {
  name: string;
  attention: number;
  blurred_confidence: number;
  drop: number;
  hint: string | null;
};

export type Analysis = {
  model: string;
  verdict: "fake" | "uncertain" | "real";
  p_fake: number;
  confidence: number;
  level: "high" | "moderate" | "low";
  calibrated: boolean;
  sure_fake: number;
  face_found: boolean;
  image: string;
  image_caption: string;
  heat_fake: string;
  heat_real: string;
  heatmap: string;
  summary: string[];
  regions: Region[];
  quality_issues: string[];
};

export type FrontierRow = {
  label: string;
  kind: "ours" | "external";
  variants: Record<string, {
    n_fake: number;
    n_real: number;
    roc_auc: number;
    detected_at_1pct_fp: number;
    fakes_caught: number;
    reals_kept: number;
  }>;
};

export type Benchmarks = {
  test: {
    id: string;
    label: string;
    short: string;
    n_images: number;
    accuracy: number;
    roc_auc: number;
    openfake_fakes: number | null;
    openfake_reals: number | null;
    unseen: Record<string, number>;
    robustness: Record<string, number> | null;
  }[];
  openfaketiny: {
    label: string;
    kind: "ours" | "external";
    fakes_caught_modern: number;
    fakes_caught_reddit: number;
    reals_kept: number;
    balanced_accuracy: number;
    roc_auc: number;
  }[];
  frontier: FrontierRow[];
};

export class ApiError extends Error {}

const UNREACHABLE =
  "Can't reach the detector API. Start it from the project folder with `python -m src.api`, then try again.";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(UNREACHABLE);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(body?.error ?? `The API answered with status ${res.status}.`);
  return body as T;
}

export async function fetchModels(): Promise<Model[]> {
  return (await request<{ models: Model[] }>("/api/models")).models;
}

export async function analyzeImage(file: File, model: string, signal?: AbortSignal): Promise<Analysis> {
  const form = new FormData();
  form.append("image", file);
  form.append("model", model);
  return request<Analysis>("/api/analyze", { method: "POST", body: form, signal });
}
