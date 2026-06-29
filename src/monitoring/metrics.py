from prometheus_client import Counter, Histogram, Gauge

INFERENCE_LATENCY = Histogram(
    "solar_inference_duration_seconds",
    "End-to-end pipeline latency per cycle",
    buckets=[1,5,10,20,30,60,90,120]
)

ALERT_COUNTER = Counter(
    "solar_alert_total",
    "Total alerts triggered",
    ["flare_class", "source"]
)

DATA_FRESHNESS = Gauge(
    "solar_data_freshness_seconds",
    "Age of most recently processed FITS file in seconds"
)

NOWCAST_CONFIDENCE = Gauge(
    "solar_nowcast_confidence",
    "Confidence of most recent nowcast prediction"
)

FAR_GAUGE = Gauge(
    "solar_far_latest",
    "False Alarm Rate from most recent walk-forward evaluation"
)

LEAD_TIME_GAUGE = Gauge(
    "solar_lead_time_minutes",
    "Mean lead time from most recent evaluation"
)
