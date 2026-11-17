import { num, pct } from "@/lib/format";

export type Column<Row> = {
  key: string;
  header: string;
  value: (row: Row) => number | null | undefined;
  format?: "pct" | "auc";
  bar?: boolean; // draw a small bar under the number
};

type Props<Row> = {
  caption: string;
  rows: Row[];
  rowLabel: (row: Row) => string;
  rowTag?: (row: Row) => string | null;
  columns: Column<Row>[];
};

/** Results table: tabular numbers, the best value per column in bold. */
export function MetricTable<Row>({ caption, rows, rowLabel, rowTag, columns }: Props<Row>) {
  const best = Object.fromEntries(
    columns.map((c) => [c.key, Math.max(...rows.map((r) => c.value(r) ?? -Infinity))]),
  );
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-fg text-xs text-muted">
            <th scope="col" className="py-2 pr-4 text-left font-medium">Detector</th>
            {columns.map((c) => (
              <th key={c.key} scope="col" className="py-2 pl-4 text-right font-medium">
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const tag = rowTag?.(row);
            return (
              <tr key={rowLabel(row)} className="border-b border-line">
                <th scope="row" className="py-3 pr-4 text-left font-medium">
                  <span className="flex flex-wrap items-center gap-2">
                    {rowLabel(row)}
                    {tag && (
                      <span className="border border-fg px-1.5 py-px text-[11px] font-medium uppercase tracking-wider">
                        {tag}
                      </span>
                    )}
                  </span>
                </th>
                {columns.map((c) => {
                  const v = c.value(row);
                  const isBest = v !== null && v !== undefined && v === best[c.key] && rows.length > 1;
                  const text = c.format === "auc" ? num(v) : pct(v);
                  return (
                    <td key={c.key} className="py-3 pl-4 text-right align-top">
                      <span className={`font-mono tabular-nums ${isBest ? "font-bold" : ""}`}>{text}</span>
                      {isBest && <span className="sr-only"> (best)</span>}
                      {c.bar && v !== null && v !== undefined && (
                        <span aria-hidden="true" className="ml-auto mt-1.5 block h-1 w-full max-w-24 bg-line">
                          <span className="block h-full bg-fg" style={{ width: `${Math.max(0, Math.min(v, 1)) * 100}%` }} />
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
