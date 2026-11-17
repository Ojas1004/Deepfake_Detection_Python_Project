"use client";

import { Check } from "@phosphor-icons/react";
import type { Model } from "@/lib/api";
import { num, pct } from "@/lib/format";

type Props = {
  models: Model[] | null;
  value: string | null;
  onChange: (id: string) => void;
};

export function ModelPicker({ models, value, onChange }: Props) {
  if (!models) {
    return (
      <div aria-hidden="true" className="space-y-3">
        <div className="h-4 w-24 animate-pulse bg-surface" />
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-16 animate-pulse bg-surface" />
        ))}
      </div>
    );
  }
  const selected = models.find((m) => m.id === value);
  const groups: { title: string; items: Model[] }[] = [
    { title: "Trained in this project", items: models.filter((m) => m.kind === "ours") },
    { title: "Downloaded detectors", items: models.filter((m) => m.kind === "external") },
  ];

  return (
    <div>
      <h2 className="text-xs font-semibold uppercase tracking-[0.14em]">1 · Model</h2>

      {/* Phones: a native select keeps the photo upload close to the top. */}
      <div className="mt-4 lg:hidden">
        <label htmlFor="model-select" className="text-xs text-muted">
          Detector
        </label>
        <select
          id="model-select"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
          className="mt-1 h-12 w-full cursor-pointer border border-line-strong bg-bg px-3 text-base"
        >
          {groups.map(
            (group) =>
              group.items.length > 0 && (
                <optgroup key={group.title} label={group.title}>
                  {group.items.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.short}
                      {m.id === models[0].id ? " (best)" : ""}
                    </option>
                  ))}
                </optgroup>
              ),
          )}
        </select>
        {selected && (
          <details className="mt-2 text-sm">
            <summary className="cursor-pointer py-2 font-medium">About this model</summary>
            <ModelFacts model={selected} />
          </details>
        )}
      </div>

      <div className="mt-4 hidden space-y-6 lg:block">
        {groups.map(
          (group) =>
            group.items.length > 0 && (
              <fieldset key={group.title}>
                <legend className="mb-2 text-xs text-muted">{group.title}</legend>
                <div className="divide-y divide-line border border-line">
                  {group.items.map((m) => {
                    const checked = m.id === value;
                    const auc = m.benchmarks.frontier?.roc_auc;
                    return (
                      <label
                        key={m.id}
                        className={`relative flex min-h-14 cursor-pointer items-start gap-3 px-3 py-3 transition-colors duration-150 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-fg ${
                          checked ? "bg-inverse text-on-inverse" : "hover:bg-surface"
                        }`}
                      >
                        <input
                          type="radio"
                          name="model"
                          value={m.id}
                          checked={checked}
                          onChange={() => onChange(m.id)}
                          className="sr-only"
                        />
                        <span
                          aria-hidden="true"
                          className={`mt-0.5 grid size-5 shrink-0 place-items-center border ${
                            checked ? "border-on-inverse" : "border-line-strong"
                          }`}
                        >
                          {checked && <Check weight="bold" className="size-3.5" />}
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="flex items-baseline justify-between gap-2">
                            <span className="text-sm font-semibold">{m.short}</span>
                            {m.id === models[0].id && (
                              <span
                                className={`shrink-0 border px-1.5 py-0.5 text-[11px] font-medium uppercase tracking-wider ${
                                  checked ? "border-on-inverse" : "border-fg"
                                }`}
                              >
                                Best
                              </span>
                            )}
                          </span>
                          <span className={`mt-0.5 block text-xs ${checked ? "opacity-80" : "text-muted"}`}>
                            {auc !== undefined ? (
                              <>
                                Frontier ROC-AUC <span className="font-mono tabular-nums">{num(auc)}</span>
                              </>
                            ) : (
                              "Not measured on the benchmarks"
                            )}
                          </span>
                        </span>
                      </label>
                    );
                  })}
                </div>
              </fieldset>
            ),
        )}
      </div>

      {selected && (
        <div className="hidden lg:block">
          <ModelFacts model={selected} />
        </div>
      )}
    </div>
  );
}

function ModelFacts({ model }: { model: Model }) {
  const f = model.benchmarks.frontier;
  const o = model.benchmarks.openfaketiny;
  return (
    <div className="mt-6 border-t border-line pt-5 text-sm">
      <p className="font-semibold">{model.label}</p>
      <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
        <dt className="text-muted">Looks at</dt>
        <dd>{model.input}</dd>
        <dt className="text-muted">Says FAKE at</dt>
        <dd className="font-mono tabular-nums">P(fake) ≥ {model.sure_fake}</dd>
        <dt className="text-muted">Heatmap</dt>
        <dd>{model.heatmap}</dd>
        {model.test && (
          <>
            <dt className="text-muted">Test accuracy</dt>
            <dd className="font-mono tabular-nums">
              {pct(model.test.accuracy)}{" "}
              <span className="font-sans text-muted">of {model.test.n_images.toLocaleString()}</span>
            </dd>
          </>
        )}
        {f && (
          <>
            <dt className="text-muted">Frontier faces</dt>
            <dd>
              <span className="font-mono tabular-nums">{pct(f.fakes_caught, 0)}</span> fakes caught,{" "}
              <span className="font-mono tabular-nums">{pct(f.reals_kept, 0)}</span> reals kept
            </dd>
          </>
        )}
        {o && (
          <>
            <dt className="text-muted">OpenFakeTiny</dt>
            <dd>
              <span className="font-mono tabular-nums">{pct(o.fakes_caught, 0)}</span> fakes caught,{" "}
              <span className="font-mono tabular-nums">{pct(o.reals_kept, 0)}</span> reals kept
            </dd>
          </>
        )}
      </dl>
      {model.note && <p className="mt-3 text-xs leading-relaxed text-muted">{model.note}</p>}
    </div>
  );
}
