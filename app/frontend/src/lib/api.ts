import type {
  AreaRow,
  MapData,
  MetricsData,
  OptionsResponse,
  PredictRequest,
  PredictResponse,
} from "./types";

// Same-origin in production; Vite proxies /api to the backend in dev.
const BASE = "/api";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  options: () => getJSON<OptionsResponse>("/options"),
  areas: () => getJSON<AreaRow[]>("/areas"),
  map: () => getJSON<MapData>("/map"),
  metrics: () => getJSON<MetricsData>("/metrics"),
  predict: async (body: PredictRequest): Promise<PredictResponse> => {
    const res = await fetch(`${BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const data = await res.json();
        detail = data.detail ?? detail;
      } catch {
        /* keep statusText */
      }
      throw new Error(detail);
    }
    return res.json() as Promise<PredictResponse>;
  },
  figureUrl: (name: string) => `${BASE}/figures/${name}`,
};
