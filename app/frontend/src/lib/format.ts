// Small presentation helpers shared across tabs.

export const pct = (v: number | null | undefined, digits = 1): string =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 0): string =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

export function minutesToHuman(min: number | null | undefined): string {
  if (min == null || Number.isNaN(min)) return "—";
  if (min < 60) return `${Math.round(min)} min`;
  if (min < 1440) {
    const h = Math.floor(min / 60);
    const m = Math.round(min % 60);
    return m ? `${h}h ${m}m` : `${h}h`;
  }
  const d = Math.floor(min / 1440);
  const h = Math.round((min % 1440) / 60);
  return h ? `${d}d ${h}h` : `${d}d`;
}

export const titleCase = (s: string | null | undefined): string =>
  !s
    ? "—"
    : s
        .replace(/[_-]+/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())
        .trim();

// Consistent semantic colours for the three manpower / risk levels.
export const LEVEL_STYLES: Record<string, { text: string; bg: string; ring: string; dot: string }> = {
  high: {
    text: "text-danger",
    bg: "bg-danger/10",
    ring: "ring-danger/30",
    dot: "bg-danger",
  },
  medium: {
    text: "text-warning",
    bg: "bg-warning/10",
    ring: "ring-warning/30",
    dot: "bg-warning",
  },
  low: {
    text: "text-success",
    bg: "bg-success/10",
    ring: "ring-success/30",
    dot: "bg-success",
  },
};

export function riskLevel(score: number): "high" | "medium" | "low" {
  if (score >= 35) return "high";
  if (score >= 18) return "medium";
  return "low";
}
