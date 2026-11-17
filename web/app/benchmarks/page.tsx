import type { Metadata } from "next";
import { API_URL, type Benchmarks } from "@/lib/api";
import { MetricTable } from "@/components/benchmarks/MetricTable";
import { pct } from "@/lib/format";

export const metadata: Metadata = { title: "Benchmarks" };
export const dynamic = "force-dynamic";

const SERVER_API = (process.env.API_URL ?? API_URL).replace(/\/$/, "");

async function load(): Promise<Benchmarks | null> {
  try {
    const res = await fetch(`${SERVER_API}/api/benchmarks`, { cache: "no-store" });
    return res.ok ? ((await res.json()) as Benchmarks) : null;
  } catch {
    return null;
  }
}

const tag = (row: { kind?: string }) => (row.kind === "ours" ? "Ours" : null);
const SOCIAL = [
  ["whatsapp", "WhatsApp"],
  ["instagram", "Instagram"],
  ["reshared", "Re-shared"],
] as const;

export default async function BenchmarksPage() {
  const data = await load();

  return (
    <>
      <div className="max-w-2xl">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">Benchmarks</h1>
        <p className="mt-4 text-base leading-relaxed text-muted">
          How every detector does on fakes from current generators it never saw in training, and on this
          project’s own test set. The best value in each column is in bold.
        </p>
      </div>

      {!data ? (
        <div role="alert" className="mt-10 border border-fg p-5 text-sm leading-relaxed">
          <p className="font-semibold">The results could not be loaded</p>
          <p className="mt-2 text-muted">
            The detector API is not running. Start it from the project folder with{" "}
            <code className="font-mono text-fg">python -m src.api</code>, then reload this page.
          </p>
        </div>
      ) : (
        <div className="mt-12 space-y-16">
          <Section
            id="frontier"
            title="Current generators: frontier benchmark"
            intro="Fakes from 19 generators released in 2025–26 (GPT-Image-2, Nano Banana Pro, FLUX.3, Seedream 5, …) and real photos. Face images only: 88 fake, 51 real. “@ 1% flagged” is the share of fakes caught when only 1% of real photos may be flagged."
          >
            <MetricTable
              caption="Frontier benchmark, original face images"
              rows={data.frontier}
              rowLabel={(r) => r.label}
              rowTag={tag}
              columns={[
                { key: "auc", header: "ROC-AUC", value: (r) => r.variants.original.roc_auc, format: "auc" },
                { key: "fp1", header: "Caught @ 1% flagged", value: (r) => r.variants.original.detected_at_1pct_fp, bar: true },
                { key: "caught", header: "Fakes caught", value: (r) => r.variants.original.fakes_caught, bar: true },
                { key: "kept", header: "Reals kept", value: (r) => r.variants.original.reals_kept, bar: true },
              ]}
            />
            <h3 className="mt-10 text-sm font-semibold">After sharing on social media</h3>
            <p className="mt-1 max-w-prose text-xs leading-relaxed text-muted">
              The same images after WhatsApp, Instagram and repeated re-sharing. Each cell: fakes caught / real photos
              kept, at threshold 0.5. Re-compression erases much of the evidence for every detector.
            </p>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full min-w-[560px] border-collapse text-sm">
                <caption className="sr-only">Frontier benchmark after social-media sharing</caption>
                <thead>
                  <tr className="border-b border-fg text-xs text-muted">
                    <th scope="col" className="py-2 pr-4 text-left font-medium">Detector</th>
                    {SOCIAL.map(([, name]) => (
                      <th key={name} scope="col" className="py-2 pl-4 text-right font-medium">{name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.frontier.map((r) => (
                    <tr key={r.label} className="border-b border-line">
                      <th scope="row" className="py-3 pr-4 text-left font-medium">{r.label}</th>
                      {SOCIAL.map(([key]) => {
                        const v = r.variants[key];
                        return (
                          <td key={key} className="py-3 pl-4 text-right font-mono tabular-nums">
                            {v ? `${pct(v.fakes_caught, 0)} / ${pct(v.reals_kept, 0)}` : "–"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section
            id="openfaketiny"
            title="Current generators: OpenFakeTiny"
            intro="850 face images: 297 fakes from current generators (GPT-image-1.5, Midjourney 7, FLUX.2, …), 488 AI images posted on Reddit, and 65 real photos. Only 65 real photos, so “reals kept” can be off by about ±10 points."
          >
            <MetricTable
              caption="OpenFakeTiny results"
              rows={data.openfaketiny}
              rowLabel={(r) => r.label}
              rowTag={tag}
              columns={[
                { key: "auc", header: "ROC-AUC", value: (r) => r.roc_auc, format: "auc" },
                { key: "bal", header: "Balanced accuracy", value: (r) => r.balanced_accuracy, bar: true },
                { key: "modern", header: "Modern fakes caught", value: (r) => r.fakes_caught_modern },
                { key: "reddit", header: "Reddit fakes caught", value: (r) => r.fakes_caught_reddit },
                { key: "kept", header: "Reals kept", value: (r) => r.reals_kept },
              ]}
            />
            <p className="mt-3 max-w-prose text-xs leading-relaxed text-muted">
              A detector that calls everything fake catches every fake but keeps no real photos; balanced accuracy and
              ROC-AUC reward getting both right.
            </p>
          </Section>

          <Section
            id="test-set"
            title="This project’s test set"
            intro={`${data.test[0]?.n_images.toLocaleString() ?? "6,900"} aligned faces, half real and half fake, from the same sources as the training data. DALL-E 3 and StyleGAN3 never appear in training.`}
          >
            <MetricTable
              caption="Results on the project's own test set"
              rows={data.test}
              rowLabel={(r) => r.short}
              columns={[
                { key: "acc", header: "Accuracy", value: (r) => r.accuracy, bar: true },
                { key: "auc", header: "ROC-AUC", value: (r) => r.roc_auc, format: "auc" },
                { key: "off", header: "OpenFake fakes", value: (r) => r.openfake_fakes },
                { key: "ofr", header: "OpenFake reals", value: (r) => r.openfake_reals },
                { key: "sg3", header: "StyleGAN3 (unseen)", value: (r) => r.unseen.stylegan3 },
                { key: "dalle", header: "DALL-E 3 (unseen)", value: (r) => r.unseen.dalle3 },
              ]}
            />
          </Section>

          <RobustnessSection rows={data.test} />
        </div>
      )}
    </>
  );
}

function Section({ id, title, intro, children }: { id: string; title: string; intro: string; children: React.ReactNode }) {
  return (
    <section aria-labelledby={`${id}-title`}>
      <h2 id={`${id}-title`} className="text-2xl font-bold tracking-tight">{title}</h2>
      <p className="mt-2 mb-6 max-w-prose text-sm leading-relaxed text-muted">{intro}</p>
      {children}
    </section>
  );
}

function RobustnessSection({ rows }: { rows: Benchmarks["test"] }) {
  const runs = rows.filter((r) => r.robustness);
  if (runs.length === 0) return null;
  const conditions = Object.keys(runs[0].robustness!);
  return (
    <Section
      id="robustness"
      title="Robustness to damaged images"
      intro="Test-set accuracy after each kind of real-world damage. Rows are conditions, columns are models."
    >
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-sm">
          <caption className="sr-only">Accuracy under image degradations</caption>
          <thead>
            <tr className="border-b border-fg text-xs text-muted">
              <th scope="col" className="py-2 pr-4 text-left font-medium">Condition</th>
              {runs.map((r) => (
                <th key={r.id} scope="col" className="py-2 pl-4 text-right font-medium">{r.short}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {conditions.map((c) => {
              const best = Math.max(...runs.map((r) => r.robustness![c]));
              return (
                <tr key={c} className="border-b border-line">
                  <th scope="row" className="py-3 pr-4 text-left font-medium">{c}</th>
                  {runs.map((r) => {
                    const v = r.robustness![c];
                    return (
                      <td key={r.id} className={`py-3 pl-4 text-right font-mono tabular-nums ${v === best ? "font-bold" : ""}`}>
                        {pct(v)}
                        {v === best && <span className="sr-only"> (best)</span>}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Section>
  );
}
