"""
Merge per-mode probe_scores_*.csv into probe_scores.csv, then run baselines,
cross-mode transfer, geometry, and all plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import directly from probe.py
from probe import (
    DATA, RESULTS, FIGURES, CACHE,
    run_tfidf_baselines, run_cross_transfer, run_geometry, run_inter_mode_cosine,
    make_plots,
)


def main():
    # Merge probe scores per mode
    parts = []
    for p in sorted(RESULTS.glob("probe_scores_*.csv")):
        df = pd.read_csv(p)
        # The mode column is already present
        parts.append(df)
        print(f"  + {p.name}  rows={len(df)}", file=sys.stderr)
    if not parts:
        print("No probe_scores_*.csv found.", file=sys.stderr)
        sys.exit(1)
    scores = pd.concat(parts, ignore_index=True)
    scores.to_csv(RESULTS / "probe_scores.csv", index=False)
    print(f"Wrote {RESULTS/'probe_scores.csv'}  rows={len(scores)}", file=sys.stderr)

    data = np.load(CACHE / "embeddings.npz", allow_pickle=True)
    all_modes = list(data["modes"])
    Ns = list(int(n) for n in data["Ns"])
    layers = list(int(L) for L in data["layers"])
    layers_used = layers[::2]

    print("\nRunning TF-IDF baselines...", file=sys.stderr)
    base_df = run_tfidf_baselines(all_modes)
    base_df.to_csv(RESULTS / "baselines.csv", index=False)

    print("\nRunning cross-mode transfer...", file=sys.stderr)
    binary_modes = [m for m in all_modes if len(np.unique(data[f"y__{m}"])) == 2]
    transfer_df = run_cross_transfer(data, binary_modes, layers_used, Ns)
    transfer_df.to_csv(RESULTS / "cross_transfer.csv", index=False)

    print("\nComputing geometry...", file=sys.stderr)
    geo_df = run_geometry(data, all_modes, layers_used, Ns)
    geo_df.to_csv(RESULTS / "geometry.csv", index=False)
    cos_df = run_inter_mode_cosine(data, all_modes, binary_modes, layers_used, Ns)
    cos_df.to_csv(RESULTS / "mode_direction_cosine.csv")

    print("\nMaking plots...", file=sys.stderr)
    make_plots(scores, base_df, transfer_df, cos_df, all_modes, Ns)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
