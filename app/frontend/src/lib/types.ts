// Shared types mirroring the backend JSON payloads.

export interface PredictRequest {
  latitude: number;
  longitude: number;
  start_datetime: string;
  event_cause?: string;
  event_type?: string;
  description?: string;
  police_station?: string;
  corridor?: string;
  zone?: string;
  junction?: string;
  direction?: string;
  veh_type?: string;
  reason_breakdown?: string;
  cargo_material?: string;
  authenticated?: string;
  address?: string;
  age_of_truck?: number | null;
  created_date?: string;
}

export interface PredictResponse {
  manpower_level: "high" | "medium" | "low";
  manpower_tier: string;
  officers_suggested: number;
  closure_probability: number;
  closure_expected: boolean;
  high_priority_probability: number;
  expected_duration_min: number;
  duration_low_min: number;
  duration_high_min: number;
  hotspot_risk: number | null;
  hotspot_flag: number;
  barricading: string;
  diversion: string;
  rationale: string;
  event_cause?: string;
  event_type?: string;
  address?: string;
}

export interface StationCentroid {
  name: string;
  lat: number;
  lng: number;
  n_events: number;
}

export interface MapBounds {
  center_lat: number;
  center_lng: number;
  min_lat: number;
  max_lat: number;
  min_lng: number;
  max_lng: number;
}

export interface OptionsResponse {
  categories: Record<string, string[]>;
  stations: StationCentroid[];
  bounds: MapBounds;
}

export interface AreaRow {
  area: string;
  n_events: number;
  lat: number;
  lng: number;
  closure_rate: number;
  pred_closure_rate: number;
  high_priority_rate: number;
  pred_high_priority_rate: number;
  avg_duration_min: number | null;
  pred_avg_duration_min: number;
  avg_hotspot_risk: number;
  chronic_count: number;
  chronic_rate: number;
  avg_officers: number;
  manpower_high: number;
  manpower_medium: number;
  manpower_low: number;
  risk_score: number;
  top_causes: string[];
}

export interface HotspotCell {
  lat: number;
  lng: number;
  count: number;
  max_risk: number;
  mean_risk: number;
  chronic_count: number;
  closure_rate: number;
  label: string;
  junction: string | null;
  top_cause: string | null;
}

export interface MapData {
  point_fields: string[];
  points: number[][];
  hotspot_cells: HotspotCell[];
  bounds: MapBounds;
}

export interface OperatingPoint {
  threshold: number;
  recall: number;
  precision: number;
  f1?: number;
  f2?: number;
  mcc: number;
}

export interface MetricsData {
  dataset: {
    n_events_scored: number;
    n_areas: number;
    closure_base_rate: number;
    high_priority_base_rate: number;
    chronic_rate: number;
    date_span: string;
  };
  priority: TaskMetrics;
  closure: TaskMetrics;
  duration: DurationMetrics;
  hotspot: HotspotMetrics;
  closure_best_operating_points: Record<string, unknown>;
}

export interface TaskMetrics {
  n: number;
  positive_rate: number;
  average_precision: number;
  ap_lift_over_base: number;
  roc_auc: number;
  f1: number;
  f_beta: number;
  precision: number;
  recall: number;
  balanced_accuracy: number;
  mcc: number;
  brier: number;
  threshold: number;
  confusion: { tn: number; fp: number; fn: number; tp: number };
  operating_points: Record<string, OperatingPoint>;
  oof?: {
    oof_ap: number;
    base_oof_ap: Record<string, number>;
  };
}

export interface DurationMetrics {
  n: number;
  mae_min: number;
  rmse_min: number;
  r2: number;
  mape: number;
  median_ae_min: number;
  mae_log: number;
  r2_log: number;
  interval_coverage_80: number;
  interval_width_med_min: number;
  n_train: number;
}

export interface HotspotMetrics {
  target: string;
  n_labelable: number;
  positive_rate: number;
  deployed_threshold: number;
  test: {
    n: number;
    base_rate: number;
    average_precision: number;
    roc_auc: number;
    brier: number;
    threshold: number;
    precision: number;
    recall: number;
    f1: number;
    mcc: number;
    precision_lift_over_base: number;
    confusion: { tn: number; fp: number; fn: number; tp: number };
  };
  test_operating_points: Record<string, OperatingPoint>;
  cold_start_test: { n: number; n_pos: number; base_rate: number };
  top_features: Record<string, number>;
}
