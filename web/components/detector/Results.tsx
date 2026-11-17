"use client";

import { CheckCircle, Question, Warning, WarningOctagon } from "@phosphor-icons/react";
import type { Analysis, Model } from "@/lib/api";
import { pct } from "@/lib/format";
import { RichText } from "@/components/RichText";
import { HeatmapViewer } from "./HeatmapViewer";
import { RegionTable } from "./RegionTable";

const VERDICT = {
  fake: {
    title: "Fake",
    line: "AI-generated, according to this model.",
    Icon: WarningOctagon,
    box: "bg-inverse text-on-inverse",
  },
  uncertain: {
    title: "Leans fake — uncertain",
    line: "Some AI-like traits, but not enough to be sure. Real photos often score in this range, especially small, heavily compressed or unusually lit ones.",
    Icon: Question,
    box: "hatch border-2 border-fg",
  },
  real: {
    title: "Real",
    line: "A genuine photograph, according to this model.",
    Icon: CheckCircle,
    box: "border-2 border-fg",
  },
} as const;

export function Results({ result, model }: { result: Analysis; model: Model | null }) {
  const v = VERDICT[result.verdict];
  const unreliable = result.quality_issues.length > 0;
  return (
    <div className="space-y-12">
      {/* 1. Verdict */}
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-[0.14em]">3 · Verdict</h2>
        <div className={`mt-4 p-6 sm:p-8 ${v.box}`}>
          <div className="flex items-start gap-4">
            <v.Icon aria-hidden="true" weight="bold" className="mt-1 size-8 shrink-0 sm:size-10" />
            <div className="min-w-0">
              <p className="text-3xl font-bold uppercase tracking-tight sm:text-5xl">{v.title}</p>
              <p className="mt-2 max-w-prose text-sm leading-relaxed sm:text-base">
                {v.line}
                {model && <> Model: {model.short}.</>}
              </p>
            </div>
          </div>
        </div>

        <dl className="mt-6 grid grid-cols-2 gap-px border border-line bg-line sm:grid-cols-3">
          <Stat label="P(fake)" value={result.p_fake.toFixed(3)} />
          <Stat
            label={result.calibrated ? "Confidence (calibrated)" : "Score for its verdict"}
            value={pct(result.confidence)}
            sub={`${result.level} confidence`}
          />
          <Stat
            label="Says FAKE at"
            value={`≥ ${result.sure_fake}`}
            sub="0.5 up to this is “uncertain”"
            className="col-span-2 sm:col-span-1"
          />
        </dl>

        <PFakeMeter p={result.p_fake} sure={result.sure_fake} />

        {(unreliable || !result.face_found) && (
          <div role="note" className="mt-6 border-l-4 border-fg bg-surface p-4 text-sm leading-relaxed">
            <p className="flex items-center gap-2 font-semibold">
              <Warning aria-hidden="true" weight="bold" className="size-5 shrink-0" />
              Don’t rely on this verdict
            </p>
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {!result.face_found && (
                <li>No face was found, so the centre of the image was used. The result is probably meaningless.</li>
              )}
              {result.quality_issues.map((issue) => (
                <li key={issue}>{issue[0].toUpperCase() + issue.slice(1)}.</li>
              ))}
            </ul>
            {unreliable && (
              <p className="mt-2 text-muted">
                Detectors look for tiny pixel-level traces of AI generation, and these conditions erase them. Try
                the original, full-size image if you have it.
              </p>
            )}
          </div>
        )}
      </div>

      {/* 2. Heatmap */}
      <HeatmapViewer key={result.heat_fake.length + result.model} result={result} />

      {/* 3. Why */}
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-[0.14em]">5 · Why</h2>
        <ul className="mt-4 max-w-prose space-y-3 text-base leading-relaxed">
          {result.summary.map((line) => (
            <li key={line} className="flex gap-3">
              <span aria-hidden="true" className="mt-2.5 size-1.5 shrink-0 bg-fg" />
              <span>
                <RichText text={line} />
              </span>
            </li>
          ))}
        </ul>
        {result.regions.length > 0 && (
          <RegionTable regions={result.regions} verdict={result.verdict} method={result.heatmap} />
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, sub, className = "" }: { label: string; value: string; sub?: string; className?: string }) {
  return (
    <div className={`bg-bg p-4 ${className}`}>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="mt-1 font-mono text-2xl font-semibold tabular-nums">{value}</dd>
      {sub && <dd className="mt-0.5 text-xs text-muted">{sub}</dd>}
    </div>
  );
}

function PFakeMeter({ p, sure }: { p: number; sure: number }) {
  return (
    <div className="mt-6">
      <div className="flex justify-between text-xs text-muted">
        <span>Real</span>
        <span>Fake</span>
      </div>
      <div
        role="meter"
        aria-label="P(fake)"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(p.toFixed(3))}
        aria-valuetext={`P(fake) ${p.toFixed(3)}; FAKE is shown from ${sure}`}
        className="relative mt-1.5 h-3 border border-fg"
      >
        <div className="h-full bg-fg" style={{ width: `${Math.max(p * 100, 0.5)}%` }} />
        <Tick at={0.5} label="0.5" />
        <Tick at={sure} label={`${sure}`} />
      </div>
      <div className="relative mt-1 h-4 text-[11px] text-muted" aria-hidden="true">
        <span className="absolute -translate-x-1/2" style={{ left: "50%" }}>
          0.5
        </span>
        <span className="absolute -translate-x-full whitespace-nowrap" style={{ left: `${Math.min(sure, 1) * 100}%` }}>
          FAKE ≥ {sure}
        </span>
      </div>
    </div>
  );
}

function Tick({ at, label }: { at: number; label: string }) {
  return (
    <span
      aria-hidden="true"
      title={label}
      className="absolute -top-1.5 h-[calc(100%+12px)] w-0.5 bg-fg mix-blend-difference"
      style={{ left: `${Math.min(at, 1) * 100}%` }}
    />
  );
}

export function ResultsSkeleton() {
  return (
    <div aria-hidden="true">
      <div className="h-3 w-24 animate-pulse bg-surface" />
      <div className="mt-4 h-36 animate-pulse bg-surface" />
      <div className="mt-6 grid grid-cols-3 gap-px">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-20 animate-pulse bg-surface" />
        ))}
      </div>
      <p className="mt-6 text-sm text-muted">
        Finding the face, running the model and testing which areas change its mind. This takes a few seconds; the
        first run of a model also loads it onto the GPU.
      </p>
      <div className="mt-10 grid gap-4 sm:grid-cols-2">
        <div className="aspect-square animate-pulse bg-surface" />
        <div className="aspect-square animate-pulse bg-surface" />
      </div>
    </div>
  );
}
