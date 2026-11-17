import type { Analysis, Region } from "@/lib/api";
import { pct, points } from "@/lib/format";

type Props = { regions: Region[]; verdict: Analysis["verdict"]; method: string };

export function RegionTable({ regions, verdict, method }: Props) {
  const word = verdict === "real" ? "REAL" : "FAKE";
  return (
    <div className="mt-8">
      <h3 className="text-sm font-semibold">Influence of each facial region</h3>
      <p className="mt-1 text-xs text-muted">
        Most influential first. “Change” is how much the confidence in {word} drops when only that region is blurred;
        positive means the region supported the verdict.
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse text-sm">
          <caption className="sr-only">Influence of each facial region on the verdict</caption>
          <thead>
            <tr className="border-b border-fg text-left text-xs text-muted">
              <th scope="col" className="py-2 pr-4 font-medium">Region</th>
              <th scope="col" className="py-2 pr-4 font-medium">{method} attention</th>
              <th scope="col" className="py-2 pr-4 text-right font-medium">Confidence if blurred</th>
              <th scope="col" className="py-2 text-right font-medium">Change</th>
            </tr>
          </thead>
          <tbody>
            {regions.map((r) => (
              <tr key={r.name} className="border-b border-line">
                <th scope="row" className="py-2.5 pr-4 text-left font-medium">{r.name}</th>
                <td className="py-2.5 pr-4">
                  <div className="flex items-center gap-3">
                    <div aria-hidden="true" className="h-1.5 w-24 bg-line">
                      <div className="h-full bg-fg" style={{ width: `${Math.min(r.attention, 1) * 100}%` }} />
                    </div>
                    <span className="font-mono text-xs tabular-nums">{pct(r.attention, 0)}</span>
                  </div>
                </td>
                <td className="py-2.5 pr-4 text-right font-mono tabular-nums">{pct(r.blurred_confidence)}</td>
                <td className={`py-2.5 text-right font-mono tabular-nums ${r.drop >= 0.05 ? "font-semibold" : ""}`}>
                  {points(r.drop)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {verdict !== "real" && (
        <details className="mt-4 text-sm">
          <summary className="cursor-pointer py-2 font-medium">What to look for in the top regions</summary>
          <p className="text-xs text-muted">General knowledge about AI-generated faces, not something the model outputs.</p>
          <ul className="mt-2 space-y-1.5">
            {regions.slice(0, 3).map(
              (r) =>
                r.hint && (
                  <li key={r.name} className="leading-relaxed">
                    <span className="font-semibold">{r.name}:</span> <span className="text-muted">{r.hint}</span>
                  </li>
                ),
            )}
          </ul>
        </details>
      )}
    </div>
  );
}
