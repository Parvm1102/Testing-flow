import { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  useMap,
  useMapEvents,
  ZoomControl,
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
import type { AreaRow, HotspotCell, MapData } from "@/lib/types";
import { HeatLayer } from "@/components/HeatLayer";
import { Spinner, ErrorState } from "@/components/ui/feedback";
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
  congestion: { 0.2: "#2f4a63", 0.45: "#3f6f93", 0.75: "#5e9cc0", 1: "#a7cfe6" },
  hotspots: { 0.2: "#4f9e7d", 0.5: "#cfa247", 0.8: "#db8a52", 1: "#d35f55" },
  closure: { 0.2: "#4f9e7d", 0.5: "#cfa247", 0.8: "#db8a52", 1: "#d35f55" },
  manpower: { 0.2: "#4f8fa0", 0.5: "#5ea585", 0.8: "#db8a52", 1: "#d35f55" },
};

function riskColor(r: number): string {
  if (r >= 0.6) return "#d35f55";
  if (r >= 0.3) return "#db8a52";
  if (r >= 0.12) return "#cfa247";
  return "#4f9e7d";
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

/**
 * Fixed pixel-radius circles render identically at every zoom, so the city
 * overview collapses into one cluttered blob. Scaling the radius with the zoom
 * level keeps the map tidy when zoomed out and legible when zoomed in.
 */
function zoomScale(zoom: number): number {
  return Math.min(Math.max(2 ** ((zoom - 13) * 0.42), 0.5), 2.1);
}

function useZoomScale(): number {
  const map = useMap();
  const [zoom, setZoom] = useState(() => map.getZoom());
  useMapEvents({ zoomend: () => setZoom(map.getZoom()) });
  return zoomScale(zoom);
}

function HotspotMarkers({ cells }: { cells: HotspotCell[] }) {
  const scale = useZoomScale();
  return (
    <>
      {cells.map((c, i) => (
        <CircleMarker
          key={i}
          center={[c.lat, c.lng]}
          radius={Math.min(2.6 + c.count * 0.4, 11) * scale}
          pathOptions={{
            color: riskColor(c.max_risk),
            fillColor: riskColor(c.max_risk),
            fillOpacity: 0.48,
            weight: 0.6,
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
    </>
  );
}

function StationMarkers({ areas }: { areas: AreaRow[] }) {
  const scale = useZoomScale();
  return (
    <>
      {areas.map((a, i) => (
        <CircleMarker
          key={`st-${i}`}
          center={[a.lat, a.lng]}
          radius={Math.min(3 + Math.sqrt(a.n_events), 13) * scale}
          pathOptions={{
            color: "#5e9cc0",
            fillColor: "#5e9cc0",
            fillOpacity: 0.14,
            weight: 1.2,
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
    </>
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

  const activeView = VIEWS.find((v) => v.id === view)!;
  // Default to Bengaluru's centre so the basemap tiles can start loading before
  // the precomputed JSON arrives — the map feels instant, data fills in after.
  const center: [number, number] = map
    ? [map.bounds.center_lat, map.bounds.center_lng]
    : [12.9716, 77.5946];

  return (
    <div>
      <div className="mb-4 flex flex-col justify-between gap-2 sm:flex-row sm:items-end">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Bengaluru risk map</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {activeView.desc}
            {map &&
              ` · ${map.points.length.toLocaleString()} events · ${map.hotspot_cells.length.toLocaleString()} hotspot cells.`}
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

        {!map && (
          <div className="absolute inset-0 z-[1100] flex items-center justify-center">
            <div className="flex items-center gap-2 rounded-full border border-border bg-card/90 px-4 py-2 text-xs font-medium text-muted-foreground shadow-sm backdrop-blur">
              <Spinner className="h-3.5 w-3.5 text-primary" /> Loading map…
            </div>
          </div>
        )}

        <MapContainer
          center={center}
          zoom={11}
          style={{ height: "100%", width: "100%" }}
          scrollWheelZoom
          preferCanvas
          zoomControl={false}
        >
          <ZoomControl position="bottomleft" />
          <TileLayer
            url={dark ? CARTO_DARK : CARTO_LIGHT}
            subdomains="abcd"
            maxZoom={19}
            attribution='&copy; OpenStreetMap &copy; CARTO'
          />
          {map && <FitBounds bounds={map.bounds} />}
          {map && <RecenterButton bounds={map.bounds} />}

          {/* Heat views */}
          {map && view !== "hotspots" && (
            <HeatLayer
              points={heatPoints}
              gradient={GRADIENTS[view]}
              radius={view === "congestion" ? 13 : 17}
              blur={15}
              max={1}
            />
          )}

          {/* Hotspot cells — radius scales with zoom to stay legible */}
          {map && view === "hotspots" && (
            <HotspotMarkers cells={map.hotspot_cells.slice(0, 700)} />
          )}

          {/* Station overlay */}
          {map && showStations && (
            <StationMarkers
              areas={areas.filter((a) => a.area !== "Unknown / No Station")}
            />
          )}
        </MapContainer>
      </div>
    </div>
  );
}

function Legend({ view }: { view: View }) {
  const items =
    view === "hotspots"
      ? [
          { c: "#d35f55", l: "≥60% risk" },
          { c: "#db8a52", l: "30–60%" },
          { c: "#cfa247", l: "12–30%" },
          { c: "#4f9e7d", l: "<12%" },
        ]
      : view === "manpower"
        ? [
            { c: "#d35f55", l: "High demand" },
            { c: "#db8a52", l: "Medium" },
            { c: "#5ea585", l: "Low" },
            { c: "#4f8fa0", l: "Minimal" },
          ]
        : view === "closure"
          ? [
              { c: "#d35f55", l: "High" },
              { c: "#db8a52", l: "Elevated" },
              { c: "#cfa247", l: "Some" },
              { c: "#4f9e7d", l: "Low" },
            ]
          : [
              { c: "#a7cfe6", l: "Dense" },
              { c: "#5e9cc0", l: "Busy" },
              { c: "#3f6f93", l: "Moderate" },
              { c: "#2f4a63", l: "Sparse" },
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
