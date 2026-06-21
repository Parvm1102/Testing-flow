import { useEffect, useState } from "react";
import { Activity, ShieldAlert, Clock, Flame } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { InfoTip } from "@/components/ui/popover";
import { LoadingState, ErrorState } from "@/components/ui/feedback";
import {
  MetricStat,
  ConfusionMatrix,
  HBarChart,
  OperatingPointsChart,
  ChartCard,
} from "@/components/charts";
import { api } from "@/lib/api";
import type {
  MetricsData,
  TaskMetrics,
  DurationMetrics,
  HotspotMetrics,
} from "@/lib/types";
import { pct, num, minutesToHuman, titleCase } from "@/lib/format";

function FigureImage({ name, alt }: { name: string; alt: string }) {
  const [ok, setOk] = useState(true);
  if (!ok) return null;
  return (
    <img
      src={api.figureUrl(name)}
      alt={alt}
      loading="lazy"
      onError={() => setOk(false)}
      className="w-full rounded-md border border-border bg-white"
    />
  );
}

function Hero({
  headline,
  unit,
  caption,
  badge,
}: {
  headline: string;
  unit?: string;
  caption: string;
  badge?: { text: string; variant: "success" | "warning" | "danger" | "accent" };
}) {
  return (
    <div className="flex flex-col justify-between gap-3 rounded-lg border border-border bg-card p-5 sm:flex-row sm:items-center">
      <div>
        <div className="flex items-baseline gap-2">
          <span className="text-4xl font-bold tracking-tight text-primary">
            {headline}
          </span>
          {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
        </div>
        <p className="mt-1 max-w-xl text-sm text-muted-foreground">{caption}</p>
      </div>
      {badge && <Badge variant={badge.variant} className="self-start text-sm">{badge.text}</Badge>}
    </div>
  );
}

/* ----------------------------- Priority ----------------------------- */
function PriorityPanel({ m }: { m: TaskMetrics }) {
  return (
    <div className="space-y-5">
      <Hero
        headline={m.average_precision.toFixed(3)}
        unit="PR-AUC"
        caption="The saturated, genuinely-easy task: priority is near-deterministic from location (is this on a designated priority corridor?), which is legitimately knowable at report time. High scores reflect honest geofencing, not leakage."
        badge={{ text: "Best task", variant: "success" }}
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricStat label="ROC-AUC" value={m.roc_auc.toFixed(3)} accent="text-accent" />
        <MetricStat label="F1" value={m.f1.toFixed(3)} />
        <MetricStat
          label="MCC"
          value={m.mcc.toFixed(3)}
          accent="text-success"
          info={{
            title: "Why MCC?",
            body: "Matthews Correlation balances all four confusion cells in one number that stays honest under class imbalance — unlike accuracy, which a trivial classifier can game. 0.90 here is near-perfect agreement.",
          }}
        />
        <MetricStat label="Brier" value={m.brier.toFixed(3)} info={{ title: "Brier score", body: "Mean squared error of the predicted probabilities. Lower = better calibrated. 0.04 means the probabilities are trustworthy, not just the rankings." }} />
        <MetricStat label="Precision" value={m.precision.toFixed(3)} />
        <MetricStat label="Recall" value={m.recall.toFixed(3)} />
        <MetricStat label="Balanced acc." value={m.balanced_accuracy.toFixed(3)} />
        <MetricStat label="Base rate" value={pct(m.positive_rate)} sub="High-priority share" />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <ChartCard
          title="Confusion at deployed threshold"
          info="Counts on the chronological test set (n=1,611) at the deployed decision threshold."
        >
          <ConfusionMatrix {...m.confusion} positiveLabel="High" negativeLabel="Low" />
        </ChartCard>
        {m.oof && (
          <ChartCard
            title="Base learners → stack (OOF AP)"
            info="Three decorrelated gradient-boosting models are combined by a logistic meta-learner on out-of-fold predictions. The stack matches or beats every individual base model."
          >
            <HBarChart
              data={[
                { name: "LightGBM", value: m.oof.base_oof_ap.lightgbm },
                { name: "XGBoost", value: m.oof.base_oof_ap.xgboost },
                { name: "CatBoost", value: m.oof.base_oof_ap.catboost },
                { name: "Stack", value: m.oof.oof_ap },
              ]}
              domain={[0.95, 1]}
              highlightLast
            />
          </ChartCard>
        )}
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <ChartCard title="Precision–recall & calibration">
          <FigureImage name="priority_pr_calibration.png" alt="Priority PR and calibration curves" />
        </ChartCard>
        <ChartCard title="SHAP feature impact" info="Which features move the priority prediction most, and in which direction.">
          <FigureImage name="priority_shap_summary.png" alt="Priority SHAP summary" />
        </ChartCard>
      </div>
    </div>
  );
}

/* ----------------------------- Closure ----------------------------- */
function ClosurePanel({ m }: { m: TaskMetrics }) {
  const ops = m.operating_points;
  const opData = [
    { key: "mcc_optimal", name: "Balanced" },
    { key: "f2_optimal", name: "Deployed (F2)" },
    { key: "recall>=0.8", name: "Never-miss" },
  ]
    .filter((o) => ops[o.key])
    .map((o) => ({
      name: o.name,
      recall: ops[o.key].recall,
      precision: ops[o.key].precision,
      mcc: ops[o.key].mcc,
    }));

  return (
    <div className="space-y-5">
      <Hero
        headline={m.average_precision.toFixed(3)}
        unit="PR-AUC"
        caption="The genuinely hard operational task. End-point/route geometry — the strongest raw predictor — was removed as leakage, so the model must forecast from real report-time signal."
        badge={{ text: `${m.ap_lift_over_base.toFixed(1)}× base rate`, variant: "accent" }}
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricStat
          label="PR-AUC lift"
          value={`${m.ap_lift_over_base.toFixed(1)}×`}
          accent="text-primary"
          info={{
            title: "Why 0.326 is strong",
            body: `With only ${pct(m.positive_rate)} of events needing a closure, a random model scores ~0.072 PR-AUC. 0.326 is ${m.ap_lift_over_base.toFixed(1)}× that base rate — for a rare, partly-discretionary event with the leakage features removed, this is substantial real signal.`,
          }}
        />
        <MetricStat label="ROC-AUC" value={m.roc_auc.toFixed(3)} accent="text-accent" />
        <MetricStat
          label="MCC"
          value={m.mcc.toFixed(3)}
          info={{ title: "Why MCC?", body: "Under 7% positives, accuracy is meaningless (predict 'never' → 93% accurate, 0 closures caught). MCC and PR-AUC reward actually finding the rare closures." }}
        />
        <MetricStat label="Brier" value={m.brier.toFixed(3)} sub="well calibrated" />
        <MetricStat label="Recall" value={pct(m.recall)} accent="text-success" sub="of real closures caught" />
        <MetricStat label="Precision" value={pct(m.precision)} sub="at deployed threshold" />
        <MetricStat
          label="F2"
          value={m.f_beta.toFixed(3)}
          info={{ title: "Why F2 (recall-leaning)?", body: "Missing a real closure (no barricade planned) is far costlier than a false alarm (an officer briefly on stand-by). F2 weights recall 2× precision, so the deployed threshold favours catching closures." }}
        />
        <MetricStat label="Base rate" value={pct(m.positive_rate)} sub="closure share" />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <ChartCard
          title="The threshold is a policy choice"
          info="The same calibrated model, scored at three operating points. Slide toward recall to never miss a closure, or toward balance for the best MCC. The demo deploys the F2 (recall-leaning) point."
          legend={[
            { color: "hsl(var(--primary))", label: "Recall" },
            { color: "hsl(var(--accent))", label: "Precision" },
            { color: "hsl(var(--warning))", label: "MCC" },
          ]}
        >
          <OperatingPointsChart data={opData} />
        </ChartCard>
        <ChartCard title="Confusion at deployed threshold" info={`Deployed F2 threshold ${m.threshold.toFixed(2)} on the test set (n=${m.n.toLocaleString()}).`}>
          <ConfusionMatrix {...m.confusion} positiveLabel="Closure" negativeLabel="None" />
        </ChartCard>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {m.oof && (
          <ChartCard title="Base learners → stack (OOF AP)" info="Out-of-fold average precision of each base model and the stacked ensemble.">
            <HBarChart
              data={[
                { name: "LightGBM", value: m.oof.base_oof_ap.lightgbm },
                { name: "XGBoost", value: m.oof.base_oof_ap.xgboost },
                { name: "CatBoost", value: m.oof.base_oof_ap.catboost },
                { name: "Stack", value: m.oof.oof_ap },
              ]}
              domain={[0.35, 0.42]}
              highlightLast
            />
          </ChartCard>
        )}
        <ChartCard title="Precision–recall & calibration">
          <FigureImage name="closure_pr_calibration.png" alt="Closure PR and calibration curves" />
        </ChartCard>
      </div>

      <ChartCard title="SHAP feature impact" info="Causal target-rate features (past-only closure rates per corridor/junction) and text embeddings carry most of the signal.">
        <div className="mx-auto max-w-2xl">
          <FigureImage name="closure_shap_summary.png" alt="Closure SHAP summary" />
        </div>
      </ChartCard>
    </div>
  );
}

/* ----------------------------- Duration ----------------------------- */
function DurationPanel({ m }: { m: DurationMetrics }) {
  return (
    <div className="space-y-5">
      <Hero
        headline={m.r2_log.toFixed(3)}
        unit="log-scale R²"
        caption="Minutes from report to clearance — a heavy-tailed target spanning minutes to multi-week construction. The honest fit is measured on the log scale; the 74-minute median error is what a control room feels on normal incidents."
        badge={{ text: `median error ${num(m.median_ae_min)} min`, variant: "warning" }}
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricStat
          label="Median abs. error"
          value={`${num(m.median_ae_min)} min`}
          accent="text-warning"
          info={{ title: "Typical incident error", body: "Half of all events are predicted within this many minutes. Robust to the multi-week outliers that distort the mean." }}
        />
        <MetricStat label="MAE (log)" value={m.mae_log.toFixed(2)} sub="central-tendency error" />
        <MetricStat label="MAPE" value={`${num(m.mape)}%`} />
        <MetricStat
          label="R² (raw min)"
          value={m.r2.toFixed(3)}
          info={{ title: "Why two R² numbers?", body: "Raw-minute R² (0.09) is dominated by a handful of days-to-weeks construction events. The log-scale R² (0.25) is the honest measure of central-tendency fit on normal incidents." }}
        />
        <MetricStat
          label="80% interval coverage"
          value={pct(m.interval_coverage_80)}
          accent="text-success"
          info={{ title: "Calibrated uncertainty", body: "The conformalised 80% prediction interval actually contains the true clearance time ~78% of the time — close to the nominal 80%. You get an honest band, not just a point." }}
        />
        <MetricStat label="Median interval width" value={`${num(m.interval_width_med_min)} min`} sub="actionable band" />
        <MetricStat label="Test events" value={m.n.toLocaleString()} sub={`train ${m.n_train.toLocaleString()}`} />
        <MetricStat label="RMSE (raw)" value={`${num(m.rmse_min)} min`} sub="inflated by tail" />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <ChartCard
          title="Why the log scale"
          info="The target spans 4+ orders of magnitude. Modelling log1p(minutes) stops a few multi-week events from dominating the loss, and is why log-R² (0.25) is the metric to trust over raw-minute R² (0.09)."
        >
          <div className="flex h-[220px] flex-col justify-center gap-4">
            <ScaleBar label="Log-scale R² (honest)" value={m.r2_log} max={0.4} color="hsl(var(--primary))" />
            <ScaleBar label="Raw-minute R² (tail-dominated)" value={m.r2} max={0.4} color="hsl(var(--muted-foreground))" />
            <ScaleBar label="80% interval coverage" value={m.interval_coverage_80} max={1} color="hsl(var(--success))" reference={0.8} />
          </div>
        </ChartCard>
        <ChartCard
          title="Prediction interval"
          info="Quantile models (p10/p50/p90) with a conformal correction produce an 80% interval around every point estimate."
        >
          <div className="flex h-[220px] flex-col items-center justify-center gap-3">
            <div className="text-center">
              <div className="text-5xl font-bold text-warning">
                {minutesToHuman(m.median_ae_min)}
              </div>
              <div className="mt-1 text-sm text-muted-foreground">median absolute error</div>
            </div>
            <div className="w-full max-w-xs">
              <div className="mb-1 flex justify-between text-[11px] text-muted-foreground">
                <span>p10</span>
                <span className="font-medium text-foreground">
                  median width {num(m.interval_width_med_min)} min
                </span>
                <span>p90</span>
              </div>
              <div className="h-2 w-full rounded-full bg-warning/25" />
            </div>
          </div>
        </ChartCard>
      </div>
    </div>
  );
}

function ScaleBar({
  label,
  value,
  max,
  color,
  reference,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
  reference?: number;
}) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-semibold tabular-nums text-foreground">{value.toFixed(3)}</span>
      </div>
      <div className="relative h-2.5 w-full rounded-full bg-secondary">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.min((value / max) * 100, 100)}%`, backgroundColor: color }}
        />
        {reference != null && (
          <div
            className="absolute top-1/2 h-3.5 w-0.5 -translate-y-1/2 bg-foreground"
            style={{ left: `${(reference / max) * 100}%` }}
            title={`nominal ${reference}`}
          />
        )}
      </div>
    </div>
  );
}

/* ----------------------------- Hotspot ----------------------------- */
function HotspotPanel({ m }: { m: HotspotMetrics }) {
  const t = m.test;
  const ops = m.test_operating_points;
  const opData = [
    { key: "mcc_optimal", name: "Balanced" },
    { key: "f2_optimal", name: "Deployed (F2)" },
  ]
    .filter((o) => ops[o.key])
    .map((o) => ({
      name: o.name,
      recall: ops[o.key].recall,
      precision: ops[o.key].precision,
      mcc: ops[o.key].mcc,
    }));

  const topFeatures = Object.entries(m.top_features)
    .slice(0, 12)
    .map(([name, value]) => ({ name: name, value }))
    .reverse();

  return (
    <div className="space-y-5">
      <Hero
        headline={t.average_precision.toFixed(3)}
        unit="PR-AUC"
        caption="A new forward-looking target engineered from scratch: will this ~110 m spot generate ≥2 more events in the next 14 days? An early warning to send a root-cause fix instead of firefighting."
        badge={{ text: `recall ${pct(t.recall)}`, variant: "success" }}
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricStat
          label="PR-AUC lift"
          value={`${(t.average_precision / t.base_rate).toFixed(1)}×`}
          accent="text-primary"
          info={{ title: "vs base rate", body: `With a ${pct(t.base_rate)} base rate, 0.441 PR-AUC is ~2.8× better than random — strong for a rare, strictly forward-looking target.` }}
        />
        <MetricStat label="ROC-AUC" value={t.roc_auc.toFixed(3)} accent="text-accent" />
        <MetricStat
          label="Recall"
          value={pct(t.recall)}
          accent="text-success"
          info={{ title: "Catch emerging hotspots", body: `At the deployed threshold the model catches ${t.confusion.tp} of ${t.confusion.tp + t.confusion.fn} emerging hotspots — only ${t.confusion.fn} missed. Early warning favours recall.` }}
        />
        <MetricStat label="Precision" value={pct(t.precision)} sub="early-warning favours recall" />
        <MetricStat label="MCC" value={t.mcc.toFixed(3)} />
        <MetricStat label="Brier" value={t.brier.toFixed(3)} />
        <MetricStat label="Base rate" value={pct(t.base_rate)} sub="chronic share" />
        <MetricStat
          label="Cold-start"
          value={pct(m.cold_start_test.base_rate)}
          sub={`${m.cold_start_test.n_pos}/${m.cold_start_test.n} new sites`}
          info={{ title: "Honest limitation", body: "Brand-new locations (no prior history) almost never turn chronic, and the model correctly assigns them low risk rather than inventing signal." }}
        />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <ChartCard title="Confusion at deployed threshold" info={`Deployed F2 threshold ${t.threshold.toFixed(3)} on the test set (n=${t.n.toLocaleString()}).`}>
          <ConfusionMatrix {...t.confusion} positiveLabel="Chronic" negativeLabel="Stable" />
        </ChartCard>
        <ChartCard
          title="Operating points"
          info="Slide from the recall-favouring deployed point to a balanced MCC-optimal point."
          legend={[
            { color: "hsl(var(--primary))", label: "Recall" },
            { color: "hsl(var(--accent))", label: "Precision" },
            { color: "hsl(var(--warning))", label: "MCC" },
          ]}
        >
          <OperatingPointsChart data={opData} />
        </ChartCard>
      </div>

      <ChartCard
        title="Top features (LightGBM gain)"
        info="Recency counts per junction / zone / police-station dominate — recurring locations are the signal. Every feature is strictly past-only (causal)."
      >
        <HBarChart
          data={topFeatures}
          color="hsl(var(--primary))"
          suffix=""
          height={320}
        />
      </ChartCard>
    </div>
  );
}

/* ----------------------------- Page ----------------------------- */
export default function Models() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.metrics().then(setMetrics).catch((e) => setError((e as Error).message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!metrics) return <LoadingState label="Loading model metrics…" />;

  return (
    <div>
      <div className="mb-5">
        <h1 className="text-xl font-bold tracking-tight">Model report</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Four leakage-controlled targets on a chronological hold-out (train on
          the past, test on the future). {metrics.dataset.n_events_scored.toLocaleString()} events ·{" "}
          {metrics.dataset.date_span}. Tap any{" "}
          <span className="font-medium text-foreground">ⓘ</span> for the “why”.
        </p>
      </div>

      <Tabs defaultValue="priority">
        <TabsList className="flex-wrap">
          <TabsTrigger value="priority">
            <Activity className="h-4 w-4" /> Priority
          </TabsTrigger>
          <TabsTrigger value="closure">
            <ShieldAlert className="h-4 w-4" /> Closure
          </TabsTrigger>
          <TabsTrigger value="duration">
            <Clock className="h-4 w-4" /> Duration
          </TabsTrigger>
          <TabsTrigger value="hotspot">
            <Flame className="h-4 w-4" /> Hotspot
          </TabsTrigger>
        </TabsList>

        <TabsContent value="priority">
          <PriorityPanel m={metrics.priority} />
        </TabsContent>
        <TabsContent value="closure">
          <ClosurePanel m={metrics.closure} />
        </TabsContent>
        <TabsContent value="duration">
          <DurationPanel m={metrics.duration} />
        </TabsContent>
        <TabsContent value="hotspot">
          <HotspotPanel m={metrics.hotspot} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
