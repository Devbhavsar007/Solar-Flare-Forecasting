import os
import time
import logging
import requests
from typing import Any

from src.ingestion.seen_files import is_seen, mark_seen
from src.monitoring.metrics import DATA_FRESHNESS
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

_SLOW_THRESHOLD_S     = 60    # SLO-2: one cycle must complete in < 60s
_SLOW_FALLBACK_S      = 120   # fallback cadence when degraded
_SLOW_ESCALATE_N      = 3     # consecutive slow cycles before Slack alert

def _post_slack(message: str) -> None:
    """Post to SLACK_WEBHOOK_URL. Silent no-op if webhook not configured."""
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook:
        logger.warning("SLACK_WEBHOOK_URL not set — Slack alert suppressed.")
        return
    try:
        r = requests.post(webhook, json={"text": message}, timeout=10)
        r.raise_for_status()
    except Exception as exc:
        logger.error(f"Slack post failed: {exc}")

def run_scheduler(pipeline: Any, pradan_downloader: Any) -> None:
    consecutive_slow = 0
    cadence_s = _SLOW_THRESHOLD_S  # start at normal cadence

    while True:
        cycle_start = time.perf_counter()
        try:
            for fits_path in pradan_downloader.list_new_files():
                instrument = "solexs" if "solexs" in fits_path.lower() else "hel1os"
                if is_seen(fits_path):           # [RULE-23]
                    logger.info(f"[RULE-23] Skip duplicate: {fits_path}")
                    continue
                state = pipeline.invoke({
                    "solexs_path": fits_path if instrument=="solexs" else "",
                    "hel1os_path": fits_path if instrument=="hel1os" else "",
                })
                mark_seen(fits_path,
                          instrument=instrument,
                          pipeline_run_id=state.get("pipeline_run_id",""))
                DATA_FRESHNESS.set(time.time() -
                                    os.path.getmtime(fits_path))  # [RULE-20]
        except Exception as exc:
            logger.error(f"Scheduler cycle error: {exc}")
        finally:
            elapsed = time.perf_counter() - cycle_start

        # SLO-2 throughput monitoring
        if elapsed > _SLOW_THRESHOLD_S:
            consecutive_slow += 1
            logger.warning(
                f"[SLO-2] Cycle {elapsed:.1f}s > {_SLOW_THRESHOLD_S}s "
                f"({consecutive_slow}/{_SLOW_ESCALATE_N})")
            if consecutive_slow >= _SLOW_ESCALATE_N:
                _post_slack(
                    f":warning: *SolarSentinel SLO-2 breach*\n"
                    f"3 consecutive cycles exceeded {_SLOW_THRESHOLD_S}s.\n"
                    f"Latest: {elapsed:.1f}s. Falling back to "
                    f"{_SLOW_FALLBACK_S}s cadence. Investigate /status endpoint.")
                cadence_s = _SLOW_FALLBACK_S   # degrade gracefully
        else:
            if consecutive_slow > 0:
                logger.info(f"[SLO-2] Recovered after {consecutive_slow} slow cycles.")
                _post_slack(":white_check_mark: *SolarSentinel SLO-2 recovered*\n"
                            f"Cycle {elapsed:.1f}s. Restoring 60s cadence.")
            consecutive_slow = 0
            cadence_s = _SLOW_THRESHOLD_S      # restore normal cadence

        sleep_for = max(0.0, cadence_s - elapsed)
        time.sleep(sleep_for)
