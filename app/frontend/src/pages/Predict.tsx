import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Users,
  ShieldAlert,
  Clock,
  Flame,
  Gauge as GaugeIcon,
  ChevronDown,
  MapPin,
  Sparkles,
  AlertTriangle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Textarea, Label } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SearchSelect } from "@/components/ui/search-select";
import { InfoTip } from "@/components/ui/popover";
import { Spinner } from "@/components/ui/feedback";
import { Gauge } from "@/components/Gauge";
import { LocationPicker } from "@/components/LocationPicker";
import { api } from "@/lib/api";
import type { OptionsResponse, PredictRequest, PredictResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import { LEVEL_STYLES, minutesToHuman, pct, titleCase } from "@/lib/format";

const DEFAULT_LAT = 12.9716;
const DEFAULT_LNG = 77.5946;
const CLOSURE_BASE = 0.072;
const PRIORITY_BASE = 0.622;
const HOTSPOT_BASE = 0.156;

function nowLocalInput(): string {
  const d = new Date();
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 16);
}

// Treat the entered wall-clock time as Bengaluru local (IST, +05:30).
const toIST = (local: string): string => `${local}:00+05:30`;

const OPTIONAL_FIELDS: { key: keyof PredictRequest; label: string; search?: boolean }[] = [
  { key: "police_station", label: "Police station", search: true },
  { key: "junction", label: "Junction", search: true },
  { key: "corridor", label: "Corridor" },
  { key: "zone", label: "Zone" },
  { key: "direction", label: "Direction" },
  { key: "veh_type", label: "Vehicle type" },
  { key: "reason_breakdown", label: "Breakdown reason" },
  { key: "cargo_material", label: "Cargo material" },
];

export default function Predict() {
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [lat, setLat] = useState(DEFAULT_LAT);
  const [lng, setLng] = useState(DEFAULT_LNG);
  const [startLocal, setStartLocal] = useState(nowLocalInput());
  const [form, setForm] = useState<Partial<PredictRequest>>({
    event_type: "unplanned",
    event_cause: "vehicle_breakdown",
    authenticated: "yes",
    description: "",
  });
  const [showOptional, setShowOptional] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.options().then(setOptions).catch(() => setOptions(null));
  }, []);

  const cats = options?.categories ?? {};

  const setField = (key: keyof PredictRequest, value: string | number | undefined) =>
    setForm((f) => ({ ...f, [key]: value }));

  async function onSubmit() {
    setLoading(true);
    setError(null);
    try {
      const payload: PredictRequest = {
        ...form,
        latitude: lat,
        longitude: lng,
        start_datetime: toIST(startLocal),
      } as PredictRequest;
      const res = await api.predict(payload);
      setResult(res);
    } catch (e) {
      setError((e as Error).message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function pickStation(name: string | undefined) {
    setField("police_station", name);
    const st = options?.stations.find((s) => s.name === name);
    if (st) {
      setLat(st.lat);
      setLng(st.lng);
    }
  }

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* ---------------- Form ---------------- */}
      <div className="xl:col-span-5">
        <div className="mb-4">
          <h1 className="text-xl font-bold tracking-tight">Forecast an event</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Enter what's known at report time. Only location and start time are
            required — everything else sharpens the forecast.
          </p>
        </div>

        <Card>
          <CardContent className="space-y-5 pt-5">
            {/* Required */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Badge variant="default" className="gap-1">
                  <Sparkles className="h-3 w-3" /> Required
                </Badge>
              </div>

              <div>
                <Label htmlFor="start">Start date &amp; time</Label>
                <Input
                  id="start"
                  type="datetime-local"
                  value={startLocal}
                  onChange={(e) => setStartLocal(e.target.value)}
                />
              </div>

              <div>
                <Label hint="quick fill">Jump to a police-station area</Label>
                <SearchSelect
                  options={(options?.stations ?? []).map((s) => s.name)}
                  value={form.police_station}
                  onChange={pickStation}
                  placeholder="Search station to center map…"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label htmlFor="lat">Latitude</Label>
                  <Input
                    id="lat"
                    type="number"
                    step="0.0001"
                    value={lat}
                    onChange={(e) => setLat(Number(e.target.value))}
                  />
                </div>
                <div>
                  <Label htmlFor="lng">Longitude</Label>
                  <Input
                    id="lng"
                    type="number"
                    step="0.0001"
                    value={lng}
                    onChange={(e) => setLng(Number(e.target.value))}
                  />
                </div>
              </div>

              <div>
                <Label hint="click to set">
                  <MapPin className="h-3 w-3" /> Location on map
                </Label>
                <LocationPicker
                  lat={lat}
                  lng={lng}
                  onPick={(la, ln) => {
                    setLat(la);
                    setLng(ln);
                  }}
                />
              </div>
            </div>

            <div className="h-px bg-border" />

            {/* Event details */}
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Event cause</Label>
                  <Select
                    value={form.event_cause}
                    onValueChange={(v) => setField("event_cause", v)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select cause" />
                    </SelectTrigger>
                    <SelectContent>
                      {(cats.event_cause ?? []).map((c) => (
                        <SelectItem key={c} value={c}>
                          {titleCase(c)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Event type</Label>
                  <Select
                    value={form.event_type}
                    onValueChange={(v) => setField("event_type", v)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select type" />
                    </SelectTrigger>
                    <SelectContent>
                      {(cats.event_type ?? ["unplanned", "planned"]).map((c) => (
                        <SelectItem key={c} value={c}>
                          {titleCase(c)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label hint="optional">
                  Description
                  <InfoTip title="Why text helps">
                    The model reads the (English/Kannada) description with a
                    multilingual sentence-transformer plus a bilingual impact
                    lexicon. Words like “diversion”, “jam”, or “ಮುಚ್ಚ” move the
                    closure and duration forecasts.
                  </InfoTip>
                </Label>
                <Textarea
                  placeholder="e.g. Lorry breakdown blocking left lane near the flyover…"
                  value={form.description}
                  onChange={(e) => setField("description", e.target.value)}
                />
              </div>
            </div>

            {/* Optional context */}
            <div>
              <button
                type="button"
                onClick={() => setShowOptional((s) => !s)}
                className="flex w-full items-center justify-between rounded-md border border-border bg-secondary/40 px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-secondary"
              >
                <span>Optional context ({OPTIONAL_FIELDS.length + 2} fields)</span>
                <ChevronDown
                  className={cn(
                    "h-4 w-4 transition-transform",
                    showOptional && "rotate-180"
                  )}
                />
              </button>

              {showOptional && (
                <div className="mt-4 grid grid-cols-2 gap-3 animate-fade-in">
                  {OPTIONAL_FIELDS.map(({ key, label, search }) => {
                    const opts = cats[key as string] ?? [];
                    return (
                      <div key={key as string}>
                        <Label>{label}</Label>
                        {search ? (
                          <SearchSelect
                            options={opts}
                            value={form[key] as string | undefined}
                            onChange={(v) => setField(key, v)}
                            placeholder="Search…"
                          />
                        ) : (
                          <Select
                            value={form[key] as string | undefined}
                            onValueChange={(v) => setField(key, v)}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Any" />
                            </SelectTrigger>
                            <SelectContent>
                              {opts.map((c) => (
                                <SelectItem key={c} value={c}>
                                  {titleCase(c)}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        )}
                      </div>
                    );
                  })}
                  <div>
                    <Label>Authenticated</Label>
                    <Select
                      value={form.authenticated}
                      onValueChange={(v) => setField("authenticated", v)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="yes" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="yes">Yes</SelectItem>
                        <SelectItem value="no">No</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Truck age (yrs)</Label>
                    <Input
                      type="number"
                      step="1"
                      min="0"
                      placeholder="—"
                      value={form.age_of_truck ?? ""}
                      onChange={(e) =>
                        setField(
                          "age_of_truck",
                          e.target.value === "" ? undefined : Number(e.target.value)
                        )
                      }
                    />
                  </div>
                </div>
              )}
            </div>

            <Button onClick={onSubmit} disabled={loading} className="w-full" size="lg">
              {loading ? <Spinner /> : <Activity className="h-4 w-4" />}
              {loading ? "Forecasting…" : "Run forecast"}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* ---------------- Results ---------------- */}
      <div className="xl:col-span-7">
        <ResultPanel result={result} loading={loading} error={error} />
      </div>
    </div>
  );
}

function ResultPanel({
  result,
  loading,
  error,
}: {
  result: PredictResponse | null;
  loading: boolean;
  error: string | null;
}) {
  if (error) {
    return (
      <Card className="flex h-full min-h-[400px] items-center justify-center">
        <div className="flex flex-col items-center gap-3 px-8 text-center">
          <AlertTriangle className="h-7 w-7 text-warning" />
          <p className="text-sm font-medium">Could not score this event</p>
          <p className="text-xs text-muted-foreground">{error}</p>
        </div>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card className="flex h-full min-h-[400px] items-center justify-center border-dashed">
        <div className="flex max-w-sm flex-col items-center gap-3 px-8 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-secondary">
            {loading ? (
              <Spinner className="h-6 w-6 text-primary" />
            ) : (
              <GaugeIcon className="h-7 w-7 text-muted-foreground" />
            )}
          </div>
          <p className="text-sm font-medium">
            {loading ? "Scoring the event…" : "Your forecast appears here"}
          </p>
          <p className="text-xs text-muted-foreground">
            Five leakage-controlled predictions plus operational recommendations,
            all from report-time inputs.
          </p>
        </div>
      </Card>
    );
  }

  return <Results result={result} />;
}

function Results({ result }: { result: PredictResponse }) {
  const level = result.manpower_level;
  const style = LEVEL_STYLES[level];

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Manpower hero */}
      <Card className={cn("relative overflow-hidden ring-1", style.ring)}>
        <div className={cn("absolute inset-0 opacity-[0.07]", style.dot)} />
        <CardContent className="relative flex flex-col items-start gap-4 pt-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            <div
              className={cn(
                "flex h-16 w-16 items-center justify-center rounded-2xl",
                style.bg
              )}
            >
              <Users className={cn("h-8 w-8", style.text)} />
            </div>
            <div>
              <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Manpower required
                <InfoTip title="How manpower is derived">
                  A transparent blend of three independent signals:
                  <span className="mt-1 block font-mono text-[11px] text-foreground">
                    1.5·closure + 1.0·priority + 0.6·duration-tier
                  </span>
                  <span className="mt-1 block">
                    It deliberately doesn't trust priority alone (priority is
                    near-deterministic from corridor), so duration and closure
                    keep it sensible. A near-certain closure always gets a crew.
                  </span>
                </InfoTip>
              </div>
              <div className={cn("text-4xl font-extrabold capitalize", style.text)}>
                {level}
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-foreground">
              {result.officers_suggested}
            </div>
            <div className="text-xs text-muted-foreground">officers suggested</div>
          </div>
        </CardContent>
      </Card>

      {/* Four predictions */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {/* Closure */}
        <Card>
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-danger" /> Road closure
              <InfoTip title="Closure probability">
                Calibrated chance this event needs a closure or diversion. Drives
                barricading. Base rate is just 7.2%, so anything well above that
                is a strong signal.
              </InfoTip>
            </CardTitle>
            <Badge variant={result.closure_expected ? "danger" : "secondary"}>
              {result.closure_expected ? "Expected" : "Unlikely"}
            </Badge>
          </CardHeader>
          <CardContent className="flex items-center justify-center">
            <Gauge
              value={result.closure_probability}
              colorClass="text-danger"
              baseline={CLOSURE_BASE}
              label={`base rate ${pct(CLOSURE_BASE, 0)}`}
            />
          </CardContent>
        </Card>

        {/* Priority */}
        <Card>
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-accent" /> High priority
              <InfoTip title="Operational priority">
                Probability the control room would tag this High priority — the
                manpower-tier driver. The model infers it from geography, cause,
                text and time (corridor is withheld to avoid a trivial lookup).
              </InfoTip>
            </CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center">
            <Gauge
              value={result.high_priority_probability}
              colorClass="text-accent"
              baseline={PRIORITY_BASE}
              label={`base rate ${pct(PRIORITY_BASE, 0)}`}
            />
          </CardContent>
        </Card>

        {/* Duration */}
        <Card>
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-warning" /> Clearance time
              <InfoTip title="Expected duration + 80% interval">
                Point estimate plus a conformalised 80% prediction interval, so
                you see the uncertainty — not just a single number.
              </InfoTip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-3xl font-bold text-warning">
                {minutesToHuman(result.expected_duration_min)}
              </div>
              <div className="mt-1 text-xs text-muted-foreground">
                expected to clear
              </div>
            </div>
            <div className="mt-4">
              <div className="mb-1 flex justify-between text-[11px] text-muted-foreground">
                <span>{minutesToHuman(result.duration_low_min)}</span>
                <span className="font-medium text-foreground">80% interval</span>
                <span>{minutesToHuman(result.duration_high_min)}</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-secondary">
                <div className="h-1.5 w-full rounded-full bg-warning/40" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Hotspot */}
        <Card>
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Flame className="h-4 w-4 text-primary" /> Chronic hotspot
              <InfoTip title="Recurring-hotspot early warning">
                Probability this ~110 m spot will reoffend (≥2 more events in 14
                days). A flag means: stop firefighting and send a root-cause fix.
              </InfoTip>
            </CardTitle>
            <Badge variant={result.hotspot_flag ? "warning" : "secondary"}>
              {result.hotspot_flag ? "Emerging" : "Stable"}
            </Badge>
          </CardHeader>
          <CardContent className="flex items-center justify-center">
            <Gauge
              value={result.hotspot_risk ?? 0}
              colorClass="text-primary"
              baseline={HOTSPOT_BASE}
              label={`base rate ${pct(HOTSPOT_BASE, 0)}`}
            />
          </CardContent>
        </Card>
      </div>

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle>Operational recommendation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <RecRow label="Barricading" value={result.barricading} />
          <RecRow label="Diversion" value={result.diversion} />
          <div className="rounded-md bg-secondary/50 p-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
            {result.rationale}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function RecRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-3">
      <div className="w-24 shrink-0 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="text-sm text-foreground">{value}</div>
    </div>
  );
}
