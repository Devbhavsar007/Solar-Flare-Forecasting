"""
JWALA FastAPI Backend — Production API Layer.

Serves health checks, Prometheus metrics, SHAP explanations,
pipeline timing/SLO status, and a real-time WebSocket feed with
strict connection cap and rate limiting [T-3].
"""
import asyncio
import hmac
import os
import time

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.utils.logging_config import configure_logging

configure_logging()
from fastapi.responses import PlainTextResponse, JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# ── Config ───────────────────────────────────────────────────
_VERSION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "version.yaml"
)
try:
    with open(_VERSION_PATH) as _f:
        _MODEL_VERSION = yaml.safe_load(_f).get("model_version", "unknown")
except FileNotFoundError:
    _MODEL_VERSION = "unknown"

_ENV = os.getenv("JWALA_ENV", "development")

# ── App ──────────────────────────────────────────────────────
app = FastAPI(title="JWALA Solar Flare Forecasting API", version=_MODEL_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket caps [T-3] ──────────────────────────────────────
MAX_CONNECTIONS = 20
_active_ws: set[WebSocket] = set()

# ── In-memory state (populated by pipeline runs) ─────────────
_latest_state: dict = {
    "timing": {},
    "last_alert": None,
    "last_explain": None,
}

# ── REST Endpoints ───────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check returning model version, environment, and staleness."""
    last_alert = _latest_state.get("last_alert")
    age_seconds = None
    if last_alert and isinstance(last_alert, dict) and "ts" in last_alert:
        try:
            from datetime import datetime, timezone
            alert_time = datetime.fromisoformat(last_alert["ts"].replace("Z", "+00:00"))
            age_seconds = (datetime.now(timezone.utc) - alert_time).total_seconds()
        except (ValueError, TypeError):
            age_seconds = None
    return {
        "status": "ok",
        "model_version": _MODEL_VERSION,
        "env": _ENV,
        "last_alert_age_seconds": age_seconds,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint [RULE-20]."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/history")
async def history(n: int = 50):
    """Return last N rows of master_catalogue.csv as JSON."""
    import pandas as pd

    catalogue_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "data", "processed", "master_catalogue.csv",
    )
    try:
        df = pd.read_csv(catalogue_path).tail(n)
        return JSONResponse(content=df.to_dict(orient="records"))
    except FileNotFoundError:
        return JSONResponse(content=[], status_code=200)


@app.get("/evaluation")
async def evaluation():
    """Return walk-forward eval metrics if available — never fabricate."""
    eval_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "data", "processed", "eval_metrics.json",
    )
    try:
        import json as _j
        with open(eval_path) as f:
            data = _j.load(f)
        data["available"] = True
        return JSONResponse(content=data)
    except (FileNotFoundError, ValueError):
        return JSONResponse(content={
            "available": False,
            "message": "No evaluation data found. Run walk_forward.py to generate eval_metrics.json."
        })


@app.get("/explain")
async def explain():
    """SHAP explanation for the last alert."""
    if _latest_state.get("last_explain") is not None:
        return _latest_state["last_explain"]
    return {"message": "No explanations available yet."}


@app.get("/status")
async def status():
    """Pipeline timing and SLO status."""
    timing = _latest_state.get("timing", {})

    # Check if any agent exceeds its budget (60s total)
    total_time = sum(timing.values()) if timing else 0
    slo_status = "PASS" if total_time <= 60 else "FAIL"
    offending = [k for k, v in timing.items() if v > 15] if timing else []

    return {
        "timing": timing,
        "total_seconds": round(total_time, 2),
        "slo_status": slo_status,
        "offending_components": offending,
    }


_NOWCAST_KEY = os.getenv("X_NOWCAST_KEY", "dev-secret-key")


@app.post("/alert")
async def alert(request: Request):
    """Broadcast a nowcast/forecast alert to all WebSocket subscribers.

    Secured with X-Nowcast-Key header using hmac.compare_digest
    to prevent timing side-channel attacks.
    """
    provided_key = request.headers.get("X-Nowcast-Key", "")
    if not hmac.compare_digest(provided_key, _NOWCAST_KEY):
        raise HTTPException(status_code=403, detail="Invalid or missing X-Nowcast-Key")

    payload = await request.json()
    _latest_state["last_alert"] = payload
    message = str(payload)
    disconnected = set()
    for ws in _active_ws.copy():
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    _active_ws.difference_update(disconnected)
    return {"status": "broadcast", "recipients": len(_active_ws)}


import json as _json
import pandas as pd

_LAST_NOWCAST: dict = {"class": "N", "confidence": 0.0}
_LAST_FORECAST: dict = {"h15": [0, 0, 0, 0]}
_LAST_ALERT: dict | None = None


def get_latest_state() -> dict:
    """Assemble the current pipeline state snapshot for WebSocket push."""
    return {
        "ts":              pd.Timestamp.utcnow().isoformat(),
        "nowcast_class":   _LAST_NOWCAST.get("class", "N"),
        "confidence":      _LAST_NOWCAST.get("confidence", 0.0),
        "forecast_15min":  _LAST_FORECAST.get("h15", [0, 0, 0, 0]),
        "alert":           _LAST_ALERT,
    }


@app.websocket("/ws/live")
async def live_feed(websocket: WebSocket):
    """
    Real-time WebSocket feed with strict controls [T-3]:
    - Maximum 20 concurrent connections
    - 1 message per 10 seconds rate limit per client
    - Outbound push on connect + every 60s cadence
    """
    if len(_active_ws) >= MAX_CONNECTIONS:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    _active_ws.add(websocket)
    last_msg_time = 0.0
    try:
        while True:
            # 1. Push current state to client every loop iteration
            await websocket.send_json(get_latest_state())

            # 2. Listen for inbound message (heartbeat/command)
            #    timeout=60 drives the push cadence: push every 60s
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=60)
                now = time.time()
                if now - last_msg_time < 10:           # rate limit [T-3]
                    await websocket.send_text(
                        _json.dumps({"error": "rate_limit",
                                    "msg":   "1 message per 10s"}))
                    await websocket.close(code=1008)
                    break
                last_msg_time = now
            except asyncio.TimeoutError:
                pass  # no inbound message — outbound already sent, loop again
    except WebSocketDisconnect:
        pass
    finally:
        _active_ws.discard(websocket)
