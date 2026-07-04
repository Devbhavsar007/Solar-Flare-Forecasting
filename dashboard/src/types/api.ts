/**
 * TypeScript interfaces for all JWALA API responses.
 */

export interface HealthResponse {
  status: string;
  model_version: string;
  env: string;
}

export interface StatusResponse {
  timing: Record<string, number>;
  total_seconds: number;
  slo_status: "PASS" | "FAIL";
  offending_components: string[];
}

export interface FlareAlert {
  flare_class: string;
  instrument: string;
  peak_flux: number;
  peak_time: string;
  confidence: number;
  start_time: string;
  end_time: string;
  forecast?: ForecastData;
  uncertainty?: UncertaintyData;
  shap?: SHAPData;
  bulletin?: string;
}

export interface ForecastData {
  prob_15min: number;
  prob_30min: number;
  prob_60min: number;
  predicted_class: number;
}

export interface UncertaintyData {
  q10: number;
  q90: number;
  median: number;
  std: number;
  uncertainty_level: string;
  mapie_set_size: number;
}

export interface SHAPData {
  top_features: [string, number][];
  class_importances: number[];
  dominant_class: number;
}

export interface FluxDataPoint {
  timestamp: string;
  sxr: number;
  hxr: number;
  q10?: number;
  q90?: number;
}

export interface HistoryRecord {
  start_time: string;
  peak_time: string;
  end_time: string;
  peak_flux: number;
  flare_class: string;
  instrument: string;
  confidence: number;
}

export interface EvaluationMetrics {
  available: boolean;
  message?: string;
  tss?: number;
  far?: number;
  tpr?: number;
  ppv?: number;
  tnr?: number;
  mean_lead_min?: number;
}