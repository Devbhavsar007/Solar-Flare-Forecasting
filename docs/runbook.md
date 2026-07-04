# SolarSentinel On-Call Runbook

## 1. solar_far_latest > 0.10 [SLO-5 breach]
   WHAT: False Alarm Rate exceeded hard limit. Model is over-alerting.
   IMMEDIATE (< 5 min):
     docker exec solar-api python -c "
       from src.monitoring.metrics import FAR_GAUGE
       print(FAR_GAUGE._value.get())"   # confirm current FAR
   DIAGNOSE (< 30 min):
     python scripts/audit_catalogue.py --last-n 50
     # Check if false alarms cluster around a specific flare class
     # or time of day (instrument noise pattern)
   FIX OPTIONS:
     A. Increase sigma_threshold in configs/nowcasting.yaml from 3.0 to 3.5
          docker compose restart worker  (no model reload needed)
     B. Rollback model: python scripts/rollback.py --run-id <prev_run_id>
          docker compose restart api worker
     C. If PRADAN data quality degraded: pause scheduler
          docker exec solar-worker kill -STOP 1  (SIGSTOP)
   ESCALATE if FAR > 0.25 for > 2 hours.

## 2. solar_lead_time_minutes < 5 [SLO-4 degradation]
   WHAT: Forecast lead time dropped - model is firing alerts too late.
   DIAGNOSE:
     docker logs solar-worker --tail 100 | grep "lead_time"
     # Check if MOMENT anomaly score dropped (no pre-flare signal)
     # Check if TimesFM is timing out (falling back to LSTM+TCN only)
   FIX:
     A. Check Ollama is running (TimesFM may have OOM-killed):
        docker compose ps | grep ollama
        docker compose restart ollama
     B. Lower forecast threshold in configs/forecasting.yaml:
        decision_threshold: 0.35   (was 0.50)   alerts fire earlier
     C. If MOMENT OOM: check memory usage. Reduce batch size in
        batch_compute_moment_scores() window stride from 60 to 120.

## 3. solar_data_freshness_seconds > 180 [SLO-2 degradation]
   WHAT: No new FITS data processed for 3+ minutes.
   DIAGNOSE:
     docker logs solar-worker --tail 50
     python scripts/download_noaa_catalog.py --start-year 2024 --end-year 2024
     #   tests network connectivity as a proxy for PRADAN access
   FIX:
     A. Check PRADAN connectivity:
        curl -sf https://pradan.issdc.gov.in/pradan/ || echo "PRADAN DOWN"
        If PRADAN is down: expected behaviour - circuit breaker is active.
        Log the outage start time. Check https://status.issdc.gov.in.
     B. Check seen_files.db isn't treating all files as already-seen:
        python -c "
          from src.ingestion.seen_files import list_seen
          import datetime
          recent = [f for f in list_seen()
                    if f['processed_at'] > str(datetime.datetime.utcnow()
                       - datetime.timedelta(hours=1))]
          print(len(recent), 'files processed in last hour')"
     C. Full restart if worker is deadlocked:
        docker compose restart worker
