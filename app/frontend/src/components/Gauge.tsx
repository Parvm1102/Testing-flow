import { cn } from "@/lib/utils";

interface GaugeProps {
  value: number; // 0..1
  label?: string;
  size?: number;
  colorClass?: string; // tailwind text-* class driving the arc colour
  baseline?: number; // optional reference line (e.g. base rate)
}

/** Compact semicircular gauge for a calibrated probability. */
export function Gauge({
  value,
  label,
  size = 132,
  colorClass = "text-primary",
  baseline,
}: GaugeProps) {
  const clamped = Math.max(0, Math.min(1, value));
  const r = size / 2 - 12;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = Math.PI * r; // semicircle
  const dash = clamped * circumference;

  const polar = (frac: number) => {
    const angle = Math.PI - frac * Math.PI; // 180deg -> 0deg
    return { x: cx + r * Math.cos(angle), y: cy - r * Math.sin(angle) };
  };
  const start = polar(0);
  const end = polar(1);
  const arc = (x: number, y: number) =>
    `M ${start.x} ${start.y} A ${r} ${r} 0 0 1 ${x} ${y}`;

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size / 2 + 16} viewBox={`0 0 ${size} ${size / 2 + 16}`}>
        {/* track */}
        <path
          d={arc(end.x, end.y)}
          fill="none"
          stroke="hsl(var(--secondary))"
          strokeWidth={10}
          strokeLinecap="round"
        />
        {/* value */}
        <path
          d={arc(end.x, end.y)}
          fill="none"
          className={colorClass}
          stroke="currentColor"
          strokeWidth={10}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference}`}
          style={{ transition: "stroke-dasharray 0.6s ease-out" }}
        />
        {/* baseline tick */}
        {baseline != null &&
          (() => {
            const p = polar(Math.max(0, Math.min(1, baseline)));
            const inner = {
              x: cx + (r - 8) * Math.cos(Math.PI - baseline * Math.PI),
              y: cy - (r - 8) * Math.sin(Math.PI - baseline * Math.PI),
            };
            return (
              <line
                x1={p.x}
                y1={p.y}
                x2={inner.x}
                y2={inner.y}
                stroke="hsl(var(--muted-foreground))"
                strokeWidth={2}
              />
            );
          })()}
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          className={cn("fill-foreground font-bold", colorClass)}
          style={{ fontSize: size * 0.2 }}
        >
          {(clamped * 100).toFixed(0)}
          <tspan style={{ fontSize: size * 0.1 }} className="fill-muted-foreground">
            %
          </tspan>
        </text>
      </svg>
      {label && (
        <span className="-mt-1 text-xs font-medium text-muted-foreground">{label}</span>
      )}
    </div>
  );
}
