import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { InfoTip } from "@/components/ui/popover";
import { cn } from "@/lib/utils";

const AXIS = "hsl(var(--muted-foreground))";
const GRID = "hsl(var(--border))";

function ChartTooltip({ active, payload, label, suffix }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-border bg-popover px-3 py-2 text-xs shadow-lg">
      <div className="font-medium text-foreground">{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="text-muted-foreground">
          {p.name}:{" "}
          <span className="font-semibold text-foreground">
            {typeof p.value === "number" ? p.value.toFixed(3) : p.value}
            {suffix}
          </span>
        </div>
      ))}
    </div>
  );
}

/** A single stat with optional explanation popover. */
export function MetricStat({
  label,
  value,
  sub,
  accent,
  info,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
  info?: { title: string; body: React.ReactNode };
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
        {info && <InfoTip title={info.title}>{info.body}</InfoTip>}
      </div>
      <div className={cn("mt-1 text-2xl font-bold tabular-nums", accent ?? "text-foreground")}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

/** 2x2 confusion matrix at the deployed threshold. */
export function ConfusionMatrix({
  tn,
  fp,
  fn,
  tp,
  positiveLabel = "Positive",
  negativeLabel = "Negative",
}: {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
  positiveLabel?: string;
  negativeLabel?: string;
}) {
  const cell = (v: number, kind: "tp" | "tn" | "fp" | "fn") => {
    const styles = {
      tp: "bg-success/15 text-success border-success/30",
      tn: "bg-success/10 text-foreground border-border",
      fp: "bg-warning/15 text-warning border-warning/30",
      fn: "bg-danger/15 text-danger border-danger/30",
    };
    const labels = { tp: "True pos", tn: "True neg", fp: "False pos", fn: "False neg" };
    return (
      <div className={cn("flex flex-col items-center justify-center rounded-md border p-3", styles[kind])}>
        <span className="text-xl font-bold tabular-nums">{v.toLocaleString()}</span>
        <span className="text-[10px] uppercase tracking-wide opacity-80">{labels[kind]}</span>
      </div>
    );
  };

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-[10px] text-muted-foreground">
        <div />
        <div className="text-center font-medium">Pred {negativeLabel}</div>
        <div className="text-center font-medium">Pred {positiveLabel}</div>
        <div className="flex items-center text-right font-medium">Actual {negativeLabel}</div>
        {cell(tn, "tn")}
        {cell(fp, "fp")}
        <div className="flex items-center text-right font-medium">Actual {positiveLabel}</div>
        {cell(fn, "fn")}
        {cell(tp, "tp")}
      </div>
    </div>
  );
}

/** Horizontal bar chart, e.g. base-learner AP or feature importance. */
export function HBarChart({
  data,
  color = "hsl(var(--primary))",
  highlightLast = false,
  domain,
  suffix = "",
  height = 220,
}: {
  data: { name: string; value: number }[];
  color?: string;
  highlightLast?: boolean;
  domain?: [number, number];
  suffix?: string;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
        <XAxis
          type="number"
          domain={domain ?? [0, "auto"]}
          tick={{ fontSize: 11, fill: AXIS }}
          stroke={GRID}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={120}
          tick={{ fontSize: 11, fill: AXIS }}
          stroke={GRID}
        />
        <Tooltip
          content={<ChartTooltip suffix={suffix} />}
          cursor={{ fill: "hsl(var(--secondary) / 0.4)" }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {data.map((_, i) => (
            <Cell
              key={i}
              fill={highlightLast && i === data.length - 1 ? "hsl(var(--accent))" : color}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Operating-point comparison as grouped bars (recall / precision / mcc). */
export function OperatingPointsChart({
  data,
}: {
  data: { name: string; recall: number; precision: number; mcc: number }[];
}) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 4 }} barGap={2}>
        <XAxis dataKey="name" tick={{ fontSize: 11, fill: AXIS }} stroke={GRID} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: AXIS }} stroke={GRID} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "hsl(var(--secondary) / 0.4)" }} />
        <Bar dataKey="recall" name="Recall" fill="hsl(var(--primary))" radius={[3, 3, 0, 0]} />
        <Bar dataKey="precision" name="Precision" fill="hsl(var(--accent))" radius={[3, 3, 0, 0]} />
        <Bar dataKey="mcc" name="MCC" fill="hsl(var(--warning))" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ChartCard({
  title,
  info,
  children,
  legend,
}: {
  title: string;
  info?: React.ReactNode;
  legend?: { color: string; label: string }[];
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold">
          {title}
          {info && <InfoTip title={title}>{info}</InfoTip>}
        </h3>
        {legend && (
          <div className="flex flex-wrap gap-3">
            {legend.map((l) => (
              <div key={l.label} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: l.color }} />
                {l.label}
              </div>
            ))}
          </div>
        )}
      </div>
      {children}
    </div>
  );
}
