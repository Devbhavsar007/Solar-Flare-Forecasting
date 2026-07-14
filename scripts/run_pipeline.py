"""
JWALA Training Pipeline Orchestrator (scripts/run_pipeline.py)

Runs every training script in the order their file dependencies actually
require, checking each prerequisite exists before launching the next step.
Stops on first failure — does not continue past a step whose output the
next step needs.

Usage:
    python scripts/run_pipeline.py                # run everything
    python scripts/run_pipeline.py --from solexs   # resume from a step
    python scripts/run_pipeline.py --only hel1os   # run a single step
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STEPS = [
    {
        "name": "goes_core",
        "cmd": [sys.executable, "scripts/train_models.py"],
        "requires": [
            "data/raw/goes",
            "data/raw/noaa_catalog.parquet",
        ],
        "produces": [
            "models/tcn_encoder.pt",
            "models/xgb_multiclass.json",
            "models/xgb_multiclass.pkl",
            "models/causal_lstm.pt",
            "models/multi_horizon.pt",
        ],
        "note": "TCN encoder pre-training -> Platt-calibrated XGBoost -> CausalLSTM -> MultiHorizonForecaster",
    },
    {
        "name": "solexs",
        "cmd": [sys.executable, "scripts/train_solexs.py"],
        "requires": [
            "data/raw/solexs/solexs_all.parquet",
            "data/raw/noaa_catalog.parquet",
        ],
        "produces": [
            "models/solexs_xgb.pkl",
            "models/solexs_xgb.json",
        ],
        "note": "Independent of goes_core. Runs any time after data is ready.",
    },
    {
        "name": "hel1os_extract",
        "cmd": [sys.executable, "scripts/extract_hel1os.py"],
        "requires": [],  # reads directly from the configured raw zip source dir
        "produces": [
            "data/raw/hel1os/hel1os_all.parquet",
        ],
        "note": "Extracts HEL1OS zips and concatenates to parquet.",
    },
    {
        "name": "hel1os_train",
        "cmd": [sys.executable, "scripts/train_hel1os.py"],
        "requires": [
            "data/raw/hel1os/hel1os_all.parquet",
            "data/raw/noaa_catalog.parquet",
        ],
        "produces": [
            "models/hel1os_binary_xgb.pkl",
        ],
        "note": "Hard-depends on hel1os_extract's output.",
    },
    {
        "name": "ensemble",
        "cmd": [sys.executable, "scripts/train_ensemble.py"],
        "requires": [
            "models/causal_lstm.pt",
            "models/multi_horizon.pt",
        ],
        "produces": [],
        "note": "Hard-depends on goes_core's CausalLSTM and MultiHorizon checkpoints.",
    },
]


def check_requires(step: dict) -> list[str]:
    missing = []
    for rel in step["requires"]:
        if not os.path.exists(os.path.join(ROOT, rel)):
            missing.append(rel)
    return missing


def check_produces(step: dict) -> list[str]:
    missing = []
    for rel in step["produces"]:
        if not os.path.exists(os.path.join(ROOT, rel)):
            missing.append(rel)
    return missing


def run_step(step: dict) -> bool:
    print(f"\n{'=' * 70}")
    print(f"STEP: {step['name']}")
    print(f"  {step['note']}")
    print(f"{'=' * 70}")

    missing = check_requires(step)
    if missing:
        print(f"ABORT: '{step['name']}' is missing required input(s):")
        for m in missing:
            print(f"    - {m}")
        print("  Fix the prerequisite step before re-running.")
        return False

    print(f"  Prerequisites OK. Running: {' '.join(step['cmd'])}\n")
    t0 = time.time()

    # Stream output live instead of capturing — per-epoch loss/F1 lines
    # from train_tcn_encoder / train_causal_lstm / train_multi_horizon
    # need to be visible as they happen, not buffered until the end.
    proc = subprocess.run(step["cmd"], cwd=ROOT)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"\nFAILED: '{step['name']}' exited with code {proc.returncode} "
              f"after {elapsed:.0f}s. Stopping pipeline.")
        return False

    still_missing = check_produces(step)
    if still_missing:
        print(f"\nWARNING: '{step['name']}' exited 0 but expected output(s) not found:")
        for m in still_missing:
            print(f"    - {m}")
        print("  Script may have silently no-op'd. Check its own log output above before trusting this step.")
        return False

    print(f"\nOK: '{step['name']}' complete in {elapsed:.0f}s. Produced:")
    for p in step["produces"]:
        print(f"    - {p}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_step", default=None,
                         help="Resume from this step name (skips earlier steps).")
    parser.add_argument("--only", dest="only_step", default=None,
                         help="Run only this single step.")
    args = parser.parse_args()

    names = [s["name"] for s in STEPS]

    if args.only_step:
        if args.only_step not in names:
            print(f"Unknown step '{args.only_step}'. Choices: {names}")
            sys.exit(1)
        steps_to_run = [s for s in STEPS if s["name"] == args.only_step]
    elif args.from_step:
        if args.from_step not in names:
            print(f"Unknown step '{args.from_step}'. Choices: {names}")
            sys.exit(1)
        start_idx = names.index(args.from_step)
        steps_to_run = STEPS[start_idx:]
    else:
        steps_to_run = STEPS

    print("JWALA Pipeline — steps to run:")
    for s in steps_to_run:
        print(f"  - {s['name']}")

    pipeline_start = time.time()
    for step in steps_to_run:
        ok = run_step(step)
        if not ok:
            sys.exit(1)

    total = time.time() - pipeline_start
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE in {total / 60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
