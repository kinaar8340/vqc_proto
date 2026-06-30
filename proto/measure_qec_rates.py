#!/usr/bin/env python3
"""Print [[3,1,3]] bit-flip QEC threshold table (γ₁ × noise_scale)."""

from __future__ import annotations

from orbital_braille.qec_stub import format_threshold_table, measure_qec_threshold


def main() -> None:
    rows = measure_qec_threshold(n_trials=10_000, seed=42)
    print(format_threshold_table(rows))


if __name__ == "__main__":
    main()