"""
Deterministic structured fallback for JWALA alert reports.

Used when the DSPy LLM module fails or times out.
Executes purely in Python with no external subprocesses or network calls.
Designed to return in < 1 second [SLO-1].
"""
import datetime

def build_alert_message(
    flare_class: str,
    peak_flux: float,
    lead_min: int,
    q10: float,
    q90: float,
    pipeline_run_id: str = "N/A",
    model_version: str = "N/A",
) -> str:
    """
    Generate a deterministic fallback alert message.

    Returns within 1 second. Maps flare class to severity text.
    """
    severity_map = {
        "X": "EXTREME",
        "M": "SEVERE",
        "C": "MODERATE",
        "B": "MINOR",
        "N": "QUIET",
    }
    severity = severity_map.get(flare_class, "UNKNOWN")

    report = (
        f"JWALA STRUCTURED ALERT: {flare_class}-Class Flare Detected\n"
        f"==========================================================\n"
        f"Severity: {severity}\n"
        f"Peak Flux: {peak_flux:.2e} W/m^2\n"
        f"Estimated Lead Time: {lead_min} minutes\n"
        f"Uncertainty Range (Q10-Q90): [{q10:.2e}, {q90:.2e}]\n\n"
        f"Pipeline Run: {pipeline_run_id}\n"
        f"Model Version: {model_version}\n\n"
        f"Note: This is an automated structured fallback report generated\n"
        f"because the natural language intelligence module was unavailable."
    )
    return report
