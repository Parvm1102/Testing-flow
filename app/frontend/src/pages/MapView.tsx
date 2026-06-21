import { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import {
  Flame,
  Waves,
  ShieldAlert,
  Users,
  Building2,
  Crosshair,
} from "lucide-react";
import { api } from "@/lib/api";
import type { AreaRow, MapData } from "@/lib/types";
import { HeatLayer } from "@/components/HeatLayer";
import { LoadingState, ErrorState } from "@/components/ui/feedback";
import { cn } from "@/lib/utils";
import { pct, minutesToHuman, titleCase, num } from "@/lib/format";

const CARTO_DARK = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
const CARTO_LIGHT =
  "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";

type View = "congestion" | "hotspots" | "closure" | "manpower";

const VIEWS: {
  id: View;
  label: string;
  icon: typeof Flame;
  desc: string;
}[] = [
  { id: "congestion", label: "Congestion", icon: Waves, desc: "Event density across the city" },
  { id: "hotspots", label: "Chronic hotspots", icon: Flame, desc: "~110 m cells likely to reoffend" },
  { id: "closure", label: "Closure risk", icon: ShieldAlert, desc: "Where closures/diversions cluster" },
  { id: "manpower", label: "Manpower load", icon: Users, desc: "Predicted officer demand" },
];

const GRADIENTS: Record<View, Record<number, string>> = {
  congestion: { 0.2: "#0ea5e9", 0.5: "#22d3ee", 0.8: "#a3e635", 1: "#facc15" },
  hotspots: { 0.2: "#16c79a", 0.5: "#facc15", 0.8: "#f97316", 1: "#ef4444" },
  closure: { 0.2: "#16c79a", 0.5: "#facc15", 0.8: "#f97316", 1: "#ef4444" },
  manpower: { 0.2: "#22d3ee", 0.5: "#a3e635", 0.8: "#f97316", 1: "#ef4444" },
};

function riskColor(r: number): string {
  if (r >= 0.6) return "#ef4444";
  if (r >= 0.3) return "#f97316";
  if (r >= 0.12) return "#facc15";
  return "#16c79a";
}

function FitBounds({ bounds }: { bounds: MapData["bounds"] }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(
      [
        [bounds.min_lat, bounds.min_lng],
        [bounds.max_lat, bounds.max_lng],
      ],
      { padding: [30, 30] }
    );
  }, [map, bounds]);
  return null;
}

function RecenterButton({ bounds }: { bounds: MapData["bounds"] }) {
  const map = useMap();
  return (
    <button
      onClick={() =>
        map.fitBounds(
          [
            [bounds.min_lat, bounds.min_lng],
            [bounds.max_lat, bounds.max_lng],
          ],
          { padding: [30, 30] }
        )
      }
      className="absolute right-3 top-3 z-[1000] flex h-9 w-9 items-center justify-center rounded-md border border-border bg-card text-muted-foreground shadow-md transition-colors hover:bg-secondary hover:text-foreground"
      title="Fit to Bengaluru"
    >
      <Crosshair className="h-4 w-4" />
    </button>
  );
}

export default function MapView() {
  const [map, setMap] = useState<MapData | null>(null);
  const [areas, setAreas] = useState<AreaRow[]>([]);
  const [view, setView] = useState<View>("hotspots");
  const [showStations, setShowStations] = useState(false);
  const [dark, setDark] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.map(), api.areas()])
      .then(([m, a]) => {
        setMap(m);
        setAreas(a);
      })
      .catch((e) => setError((e as Error).message));
  }, []);

  const heatPoints = useMemo<L.HeatLatLngTuple[]>(() => {
    if (!map || view === "hotspots") return [];
    return map.points.map((p) => {
      const [lat, lng, closure, , officers] = p;
      let w = 0.4;
      if (view === "congestion") w = 0.5;
      else if (view === "closure") w = Math.max(0.1, closure);
      else if (view === "manpower") w = Math.max(0.1, officers / 6);
      return [lat, lng, w] as L.HeatLatLngTuple;
    });
  }, [map, view]);

  if (error) return <ErrorState message={error} />;
  if (!map) return <LoadingState label="Loading map…" />;

  const activeView = VIEWS.find((v) => v.id === view)!;

  return (
    <div>
      <div className="mb-4 flex flex-col justify-between gap-2 sm:flex-row sm:items-end">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Bengaluru risk map</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {activeView.desc} · {map.points.length.toLocaleString()} events ·{" "}
            {map.hotspot_cells.length.toLocaleString()} hotspot cells.
          </p>
        </div>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={showStations}
            onChange={(e) => setShowStations(e.target.checked)}
            className="h-3.5 w-3.5 accent-primary"
          />
          <Building2 className="h-3.5 w-3.5" /> Station centers
        </label>
      </div>

      <div className="relative h-[68vh] overflow-hidden rounded-xl border border-border">
        {/* View selector */}
        <div className="absolute left-3 top-3 z-[1000] flex flex-col gap-1 rounded-lg border border-border bg-card/90 p-1.5 shadow-lg backdrop-blur">
          {VIEWS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              className={cn(
                "flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                view === id
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground"
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
          ))}
          <div className="my-1 h-px bg-border" />
          <button
            onClick={() => setDark((d) => !d)}
            className="px-3 py-1 text-[11px] text-muted-foreground hover:text-foreground"
          >
            {dark ? "Light basemap" : "Dark basemap"}
          </button>
        </div>

        <Legend view={view} />

        <MapContainer
          center={[map.bounds.center_lat, map.bounds.center_lng]}
          zoom={11}
          style={{ height: "100%", width: "100%" }}
          scrollWheelZoom
          preferCanvas
        >
          <TileLayer
            url={dark ? CARTO_DARK : CARTO_LIGHT}
            subdomains="abcd"
            maxZoom={19}
            attribution='&copy; OpenStreetMap &copy; CARTO'
          />
          <FitBounds bounds={map.bounds} />
          <RecenterButton bounds={map.bounds} />

          {/* Heat views */}
          {view !== "hotspots" && (
            <HeatLayer
              points={heatPoints}
              gradient={GRADIENTS[view]}
              radius={view === "congestion" ? 18 : 22}
              blur={16}
              max={1}
            />
          )}

          {/* Hotspot cells */}
          {view === "hotspots" &&
            map.hotspot_cells.slice(0, 700).map((c, i) => (
              <CircleMarker
                key={i}
                center={[c.lat, c.lng]}
                radius={Math.min(4 + c.count * 0.7, 16)}
                pathOptions={{
                  color: riskColor(c.max_risk),
                  fillColor: riskColor(c.max_risk),
                  fillOpacity: 0.55,
                  weight: 1,
                }}
              >
                <Popup>
                  <div className="space-y-1">
                    <div className="text-sm font-semibold">{c.label}</div>
                    {c.junction && (
                      <div className="text-xs text-muted-foreground">
                        {titleCase(c.junction)}
                      </div>
                    )}
                    <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 pt-1 text-xs">
                      <span className="text-muted-foreground">Events</span>
                      <span className="text-right font-medium">{c.count}</span>
                      <span className="text-muted-foreground">Max risk</span>
                      <span className="text-right font-medium">{pct(c.max_risk)}</span>
                      <span className="text-muted-foreground">Chronic</span>
                      <span className="text-right font-medium">{c.chronic_count}</span>
                      <span className="text-muted-foreground">Closure rate</span>
                      <span className="text-right font-medium">{pct(c.closure_rate)}</span>
                      {c.top_cause && (
                        <>
                          <span className="text-muted-foreground">Top cause</span>
                          <span className="text-right font-medium">
                            {titleCase(c.top_cause)}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            ))}

          {/* Station overlay */}
          {showStations &&
            areas
              .filter((a) => a.area !== "Unknown / No Station")
              .map((a, i) => (
                <CircleMarker
                  key={`st-${i}`}
                  center={[a.lat, a.lng]}
                  radius={Math.min(5 + Math.sqrt(a.n_events), 18)}
                  pathOptions={{
                    color: "#2bb7f0",
                    fillColor: "#2bb7f0",
                    fillOpacity: 0.18,
                    weight: 1.5,
                  }}
                >
                  <Popup>
                    <div className="space-y-1">
                      <div className="text-sm font-semibold">{a.area}</div>
                      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 pt-1 text-xs">
                        <span className="text-muted-foreground">Events</span>
                        <span className="text-right font-medium">{a.n_events}</span>
                        <span className="text-muted-foreground">Risk score</span>
                        <span className="text-right font-medium">{a.risk_score}</span>
                        <span className="text-muted-foreground">Closure rate</span>
                        <span className="text-right font-medium">{pct(a.closure_rate)}</span>
                        <span className="text-muted-foreground">Avg clear</span>
                        <span className="text-right font-medium">
                          {minutesToHuman(a.avg_duration_min)}
                        </span>
                        <span className="text-muted-foreground">Avg officers</span>
                        <span className="text-right font-medium">{num(a.avg_officers, 1)}</span>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
        </MapContainer>
      </div>
    </div>
  );
}

function Legend({ view }: { view: View }) {
  const items =
    view === "hotspots"
      ? [
          { c: "#ef4444", l: "≥60% risk" },
          { c: "#f97316", l: "30–60%" },
          { c: "#facc15", l: "12–30%" },
          { c: "#16c79a", l: "<12%" },
        ]
      : view === "manpower"
        ? [
            { c: "#ef4444", l: "High demand" },
            { c: "#f97316", l: "Medium" },
            { c: "#a3e635", l: "Low" },
            { c: "#22d3ee", l: "Minimal" },
          ]
        : view === "closure"
          ? [
              { c: "#ef4444", l: "High" },
              { c: "#f97316", l: "Elevated" },
              { c: "#facc15", l: "Some" },
              { c: "#16c79a", l: "Low" },
            ]
          : [
              { c: "#facc15", l: "Dense" },
              { c: "#a3e635", l: "Busy" },
              { c: "#22d3ee", l: "Moderate" },
              { c: "#0ea5e9", l: "Sparse" },
            ];

  return (
    <div className="absolute bottom-3 right-3 z-[1000] rounded-lg border border-border bg-card/90 p-3 shadow-lg backdrop-blur">
      <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        Legend
      </div>
      <div className="space-y-1">
        {items.map((it) => (
          <div key={it.l} className="flex items-center gap-2 text-xs">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: it.c }}
            />
            <span className="text-muted-foreground">{it.l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
