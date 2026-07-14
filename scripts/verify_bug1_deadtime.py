"""
BUG 1 Verification: dead-time correction bracket fix.
Tests correct_single at various fractions of saturation_rate.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.optimize import brentq

# Reproduce the corrected function exactly as in fits_reader.py
dead_time_us = 2.5
tau = dead_time_us * 1e-6
saturation_rate = 1.0 / (np.e * tau)
r_max_bracket = 1.0 / tau

print(f"tau = {tau} s")
print(f"saturation_rate = {saturation_rate:.4f} cts/s")
print(f"r_max_bracket (1/tau) = {r_max_bracket:.4f} cts/s")
print()

def correct_single(n_obs: float) -> float:
    rate_obs = n_obs
    if rate_obs <= 0:
        return 0.0
    if rate_obs >= saturation_rate * 0.999:
        return np.nan
    return brentq(
        lambda r: r * np.exp(-r * tau) - rate_obs,
        0, r_max_bracket,
        xtol=1e-7
    )

fractions = [0.50, 0.90, 0.95, 0.99, 0.999, 1.00, 1.10]

print(f"{'Fraction':>10} | {'rate_obs':>14} | {'corrected':>14} | {'status'}")
print("-" * 65)
for frac in fractions:
    rate_obs = frac * saturation_rate
    result = correct_single(rate_obs)
    if np.isnan(result):
        status = "NaN (-> interpolate)"
    else:
        # Verify: result * exp(-result * tau) should == rate_obs
        check = result * np.exp(-result * tau)
        status = f"brentq OK (check={check:.4f})"
    print(f"{frac:>9.1%} | {rate_obs:>14.4f} | {str(result):>14} | {status}")

# --- Unit test assertion ---
print("\n--- Unit test: 95% of saturation_rate ---")
test_val = correct_single(0.95 * saturation_rate)
assert np.isfinite(test_val), f"Expected finite, got {test_val}"
assert test_val > 0.95 * saturation_rate, f"Corrected should be > observed (got {test_val})"
print(f"PASS: correct_single(0.95 * sat_rate) = {test_val:.4f} (finite, > input)")
