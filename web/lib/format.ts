export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return value.toFixed(digits);
}

/** Signed percentage points, e.g. +12.3 pts. */
export function points(value: number): string {
  const v = value * 100;
  if (Math.abs(v) < 0.05) return "0.0 pts";
  return `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(1)} pts`;
}

export function fileSize(bytes: number): string {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
