"""Compare MATLAB validation output against the Python fixture expectations."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION = PROJECT_ROOT / "validation"

EXACT_KEYS = {"physical_mcs", "physical_n_bps", "legacy_multi_mcs", "legacy_multi_n_bps", "wall_intersections", "ilp_selected_indices", "greedy_selected_indices", "candidate_group_rows", "nsc"}

KEYS = [
    "physical_sinr_db", "physical_mcs", "physical_n_bps", "physical_coding_rate",
    "physical_ber", "physical_per", "physical_phy_rate_bps", "aligned_group_txop_s",
    "aligned_rates_bps", "aligned_weighted_reward", "conventional_rates_bps",
    "conventional_reward", "legacy_multi_sinr_db", "legacy_multi_mcs",
    "legacy_multi_n_bps", "legacy_multi_per", "legacy_multi_phy_rate_bps",
    "legacy_aligned_rates_bps", "legacy_conventional_rates_bps",
    "singleton_physical_sinr_db", "singleton_legacy_sinr_db", "pathloss_db",
    "txpower_dbm", "nsc", "csr_be", "wall_intersections",
    "ilp_selected_indices", "ilp_objective", "greedy_selected_indices", "greedy_objective",
    "candidate_group_rows",
]


def _load_vector(path: Path, key: str) -> np.ndarray:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    return np.asarray(data[key], dtype=float).reshape(-1)


def main() -> int:
    expected_path = VALIDATION / "python_expected.mat"
    actual_path = VALIDATION / "matlab_validation_output.mat"
    if not expected_path.exists() or not actual_path.exists():
        print("Missing validation files. Generate fixture, run MATLAB validator, then compare.")
        print(f"Expected: {expected_path}")
        print(f"MATLAB output: {actual_path}")
        return 2

    failed = False
    print("Comparing MATLAB outputs against Python expectations")
    print("Integer/index arrays: exact; floating-point arrays: rtol=1e-9 with key-specific absolute tolerance\n")
    for key in KEYS:
        expected = _load_vector(expected_path, key)
        actual = _load_vector(actual_path, key)
        if expected.shape != actual.shape:
            print(f"FAIL {key}: shape mismatch expected={expected.shape}, actual={actual.shape}")
            failed = True
            continue
        if not np.array_equal(np.isnan(expected), np.isnan(actual)):
            passed = False
            max_abs = float("inf")
        elif key in EXACT_KEYS:
            passed = bool(np.array_equal(expected, actual))
            max_abs = float(np.max(np.abs(expected - actual))) if expected.size else 0.0
        elif np.all(np.isnan(expected)) and np.all(np.isnan(actual)):
            max_abs = 0.0
            passed = True
        else:
            finite = np.isfinite(expected) & np.isfinite(actual)
            if finite.any():
                max_abs = float(np.max(np.abs(expected[finite] - actual[finite])))
                # Tight absolute tolerance for tiny BER/PER values; looser for
                # high-magnitude bps quantities where relative error dominates.
                atol = 1e-14 if key in {"physical_ber", "physical_per", "legacy_multi_per"} else 1e-8
                passed = bool(np.allclose(expected, actual, rtol=1e-9, atol=atol, equal_nan=True))
            else:
                passed = True
                max_abs = 0.0
        print(f"{'PASS' if passed else 'FAIL'} {key}: max_abs_diff={max_abs:.3e}")
        failed |= not passed

    if failed:
        print("\nSome values differ. Inspect dimensions, units, and source-version consistency.")
        return 1
    print("\nAll shared MATLAB/Python validation values match within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
