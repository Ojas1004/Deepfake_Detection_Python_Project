"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowClockwise, WarningCircle } from "@phosphor-icons/react";
import { ApiError, analyzeImage, fetchModels, type Analysis, type Model } from "@/lib/api";
import { ModelPicker } from "./ModelPicker";
import { UploadZone } from "./UploadZone";
import { Results, ResultsSkeleton } from "./Results";

type Status = "idle" | "loading" | "done" | "error";

export function Detector() {
  const [models, setModels] = useState<Model[] | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelId, setModelId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const resultsRef = useRef<HTMLElement>(null);

  const loadModels = useCallback(async () => {
    try {
      const list = await fetchModels();
      setModelsError(null);
      setModels(list);
      setModelId((current) => current ?? list[0]?.id ?? null);
    } catch (err) {
      setModelsError((err as Error).message);
    }
  }, []);

  useEffect(() => {
    // Subscribes to the API once on mount; state is only set when the request settles.
    loadModels(); // eslint-disable-line react-hooks/set-state-in-effect
  }, [loadModels]);

  const analyze = useCallback(async (f: File, model: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setStatus("loading");
    setError(null);
    try {
      const analysis = await analyzeImage(f, model, controller.signal);
      setResult(analysis);
      setStatus("done");
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setError(err instanceof ApiError ? err.message : "Something went wrong while analysing the image.");
      setStatus("error");
    }
  }, []);

  // Free the preview and cancel any request when leaving the page.
  const previewRef = useRef<string | null>(null);
  useEffect(
    () => () => {
      abortRef.current?.abort();
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    },
    [],
  );

  const onFile = useCallback(
    (f: File) => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
      previewRef.current = URL.createObjectURL(f);
      setPreviewUrl(previewRef.current);
      setFile(f);
      setResult(null);
      if (modelId) analyze(f, modelId);
      requestAnimationFrame(() => resultsRef.current?.focus({ preventScroll: true }));
    },
    [modelId, analyze],
  );

  const onModel = useCallback(
    (id: string) => {
      setModelId(id);
      if (file) analyze(file, id);
    },
    [file, analyze],
  );

  const model = models?.find((m) => m.id === modelId) ?? null;

  return (
    <div className="mt-10 grid gap-10 lg:grid-cols-[340px_minmax(0,1fr)] lg:gap-12">
      <aside aria-label="Model" className="lg:order-none">
        {modelsError ? (
          <div role="alert" className="border border-fg p-4 text-sm leading-relaxed">
            <p className="flex items-start gap-2 font-semibold">
              <WarningCircle aria-hidden="true" className="mt-0.5 size-5 shrink-0" />
              The detector API is not running
            </p>
            <p className="mt-2 text-muted">{modelsError}</p>
            <button
              type="button"
              onClick={() => {
                setModelsError(null);
                loadModels();
              }}
              className="mt-4 inline-flex h-11 cursor-pointer items-center gap-2 border border-fg px-4 text-sm font-medium transition-colors duration-150 hover:bg-inverse hover:text-on-inverse"
            >
              <ArrowClockwise aria-hidden="true" className="size-4" />
              Try again
            </button>
          </div>
        ) : (
          <ModelPicker models={models} value={modelId} onChange={onModel} />
        )}
      </aside>

      <div className="min-w-0">
        <UploadZone file={file} previewUrl={previewUrl} onFile={onFile} busy={status === "loading"} />

        <section
          ref={resultsRef}
          tabIndex={-1}
          aria-label="Result"
          aria-busy={status === "loading"}
          className="mt-10 outline-none"
        >
          <div aria-live="polite" className="sr-only">
            {status === "loading" && "Analysing the image."}
            {status === "done" && result && `Result: ${result.verdict === "uncertain" ? "leans fake, uncertain" : result.verdict}.`}
            {status === "error" && error}
          </div>

          {status === "loading" && <ResultsSkeleton />}

          {status === "error" && error && (
            <div role="alert" className="border border-fg p-5 text-sm leading-relaxed">
              <p className="flex items-start gap-2 font-semibold">
                <WarningCircle aria-hidden="true" className="mt-0.5 size-5 shrink-0" />
                The image could not be analysed
              </p>
              <p className="mt-2 text-muted">{error}</p>
              {file && modelId && (
                <button
                  type="button"
                  onClick={() => analyze(file, modelId)}
                  className="mt-4 inline-flex h-11 cursor-pointer items-center gap-2 border border-fg px-4 font-medium transition-colors duration-150 hover:bg-inverse hover:text-on-inverse"
                >
                  <ArrowClockwise aria-hidden="true" className="size-4" />
                  Try again
                </button>
              )}
            </div>
          )}

          {status === "done" && result && <Results result={result} model={model} />}

          {status === "idle" && (
            <p className="border-t border-line pt-6 text-sm text-muted">
              The verdict, a heatmap of the evidence and a plain-language explanation will appear here.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
