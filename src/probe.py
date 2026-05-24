"""
Train layer-wise probes for each mode × N, plus baselines and geometry analysis.

Outputs:
    results/probe_scores.csv      mode,N,layer,probe,fold,acc,auroc
    results/baselines.csv         mode,probe,acc,auroc          (TF-IDF baseline)
    results/cross_transfer.csv    train_mode,test_mode,layer,N,acc
    results/geometry.csv          mode,N,layer,intrinsic_dim,explained_var_first_pc
    figures/*.png

Faster build: per-fold StandardScaler dropped (we use one-shot scaling), and
multiclass LogReg uses lbfgs with relaxed tolerance.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5


def _scaled_auroc(y_true, scores_or_proba) -> float:
    """Multi-class one-vs-rest AUROC; falls back to binary AUROC."""
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    if len(classes) == 2:
        if scores_or_proba.ndim == 2 and scores_or_proba.shape[1] == 2:
            scores = scores_or_proba[:, 1]
        else:
            scores = scores_or_proba
        return roc_auc_score(y_true, scores)
    if scores_or_proba.ndim == 1:
        return np.nan
    try:
        return roc_auc_score(y_true, scores_or_proba, multi_class="ovr", average="macro")
    except ValueError:
        return np.nan


def _fit(X_train, y_train, n_classes: int):
    """Fast probe: binary uses liblinear, multiclass uses lbfgs with relaxed tol."""
    if n_classes == 2:
        clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=200, tol=1e-3)
    else:
        clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200, tol=1e-3)
    clf.fit(X_train, y_train)
    return clf


def probe_run(X, y, n_classes: int, pca_dim: int | None = None):
    """K-fold cross-validation with one-shot global scaling.

    For multiclass problems, an optional PCA reduction to `pca_dim` speeds up
    the LogReg solver dramatically with negligible accuracy loss (the relevant
    information lives in the top components).
    """
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
    Xs = scaler.transform(X)
    if pca_dim is not None and pca_dim < Xs.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=SEED).fit(Xs)
        Xs = pca.transform(Xs)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(Xs, y_enc)):
        clf = _fit(Xs[tr], y_enc[tr], n_classes)
        pred = clf.predict(Xs[te])
        try:
            proba = clf.predict_proba(Xs[te])
        except AttributeError:
            proba = None
        acc = accuracy_score(y_enc[te], pred)
        auroc = _scaled_auroc(y_enc[te], proba) if proba is not None else np.nan
        rows.append({"fold": fold, "acc": acc, "auroc": auroc})
    return pd.DataFrame(rows)


def run_main_sweep(data, modes, Ns, layers_used, pca_dim_multiclass: int | None = 256):
    rows = []
    for mode in modes:
        y = data[f"y__{mode}"]
        n_classes = len(np.unique(y))
        if n_classes < 2:
            print(f"  skip mode {mode} — only one class", file=sys.stderr)
            continue
        pca_dim = pca_dim_multiclass if n_classes > 2 else None
        for N in Ns:
            X_full = data[f"X__{mode}__N{N}"]  # (n, layers, dim)
            for L in layers_used:
                X = X_full[:, L, :]
                df = probe_run(X, y, n_classes, pca_dim=pca_dim)
                df["mode"] = mode; df["N"] = N; df["layer"] = L
                rows.append(df)
                print(
                    f"  mode={mode} N={N} L={L:>2} "
                    f"acc={df['acc'].mean():.3f}±{df['acc'].std():.3f} "
                    f"auroc={df['auroc'].mean():.3f}",
                    file=sys.stderr, flush=True
                )
    return pd.concat(rows, ignore_index=True)


def run_tfidf_baselines(modes):
    chunks = pd.read_parquet(DATA / "chunks.parquet")
    base_rows = []
    for mode in modes:
        sub = chunks[chunks["mode"] == mode]
        if sub["label"].nunique() < 2:
            continue
        X_text = sub["text"].tolist()
        y_text = sub["label"].to_numpy()
        vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        X_tfidf = vect.fit_transform(X_text)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        accs, aurocs = [], []
        le = LabelEncoder().fit(y_text)
        y_enc = le.transform(y_text)
        for tr, te in skf.split(X_tfidf, y_enc):
            clf = LogisticRegression(max_iter=500, solver="lbfgs", tol=1e-3)
            clf.fit(X_tfidf[tr], y_enc[tr])
            pred = clf.predict(X_tfidf[te])
            proba = clf.predict_proba(X_tfidf[te])
            accs.append(accuracy_score(y_enc[te], pred))
            aurocs.append(_scaled_auroc(y_enc[te], proba))
        base_rows.append({
            "mode": mode, "probe": "tfidf_logreg",
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "auroc_mean": float(np.nanmean(aurocs)),
            "auroc_std": float(np.nanstd(aurocs)),
        })
        print(f"  TF-IDF  mode={mode}  acc={np.mean(accs):.3f}±{np.std(accs):.3f}",
              file=sys.stderr, flush=True)
    return pd.DataFrame(base_rows)


def run_cross_transfer(data, binary_modes, layers_used, Ns):
    """Train binary probe on mode A, test direction transferability to mode B."""
    transfer_rows = []
    L_xfer = layers_used[-1]
    N_xfer = max(Ns)
    # Cache (scaled X, y, probe direction) per mode
    cache = {}
    for ma in binary_modes:
        Xa = data[f"X__{ma}__N{N_xfer}"][:, L_xfer, :]
        ya = LabelEncoder().fit_transform(data[f"y__{ma}"])
        sca = StandardScaler().fit(Xa)
        Xa_s = sca.transform(Xa)
        clf_a = LogisticRegression(solver="liblinear", max_iter=200, tol=1e-3)
        clf_a.fit(Xa_s, ya)
        w_a = clf_a.coef_.ravel()
        cache[ma] = (Xa_s, ya, w_a, sca)
    for ma in binary_modes:
        _, _, w_a, _ = cache[ma]
        for mb in binary_modes:
            Xb_s, yb, _, _ = cache[mb]
            proj_b = Xb_s @ w_a
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            accs = []
            for tr, te in skf.split(proj_b.reshape(-1, 1), yb):
                clf = LogisticRegression(solver="liblinear", max_iter=200, tol=1e-3)
                clf.fit(proj_b[tr].reshape(-1, 1), yb[tr])
                accs.append(accuracy_score(yb[te], clf.predict(proj_b[te].reshape(-1, 1))))
            transfer_rows.append({
                "train_mode": ma, "test_mode": mb,
                "layer": L_xfer, "N": N_xfer,
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
            })
            print(f"  {ma:>12s} -> {mb:<12s}  acc={np.mean(accs):.3f}",
                  file=sys.stderr, flush=True)
    return pd.DataFrame(transfer_rows)


def run_geometry(data, modes, layers_used, Ns):
    L_g = layers_used[-1]; N_g = max(Ns)
    rows = []
    for mode in modes:
        y = data[f"y__{mode}"]
        if len(np.unique(y)) < 2:
            continue
        X = data[f"X__{mode}__N{N_g}"][:, L_g, :]
        classes = np.unique(y)
        means = np.stack([X[y == c].mean(axis=0) for c in classes], axis=0)
        diffs = means - means.mean(axis=0, keepdims=True)
        n_components = max(1, len(classes) - 1)
        n_components = min(n_components, X.shape[1])
        if n_components >= 1 and len(classes) >= 2:
            pca = PCA(n_components=n_components).fit(diffs)
            cum = np.cumsum(pca.explained_variance_ratio_)
            intrinsic = int(np.searchsorted(cum, 0.95) + 1)
            explained = pca.explained_variance_ratio_.tolist()
        else:
            intrinsic = 0
            explained = [0.0]
        pca_full = PCA(n_components=min(50, X.shape[1])).fit(X)
        rows.append({
            "mode": mode, "N": N_g, "layer": L_g,
            "n_classes": int(len(classes)),
            "class_mean_subspace_dim": int(n_components),
            "intrinsic_dim_95pct": int(intrinsic),
            "first_pc_var_class": float(explained[0]),
            "first_pc_var_full": float(pca_full.explained_variance_ratio_[0]),
        })
    return pd.DataFrame(rows)


def run_inter_mode_cosine(data, modes, binary_modes, layers_used, Ns):
    L_g = layers_used[-1]; N_g = max(Ns)
    dirs = {}
    for mode in binary_modes:
        X = data[f"X__{mode}__N{N_g}"][:, L_g, :]
        y = LabelEncoder().fit_transform(data[f"y__{mode}"])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(solver="liblinear", max_iter=200, tol=1e-3)
        clf.fit(sc.transform(X), y)
        w = clf.coef_.ravel()
        dirs[mode] = w / (np.linalg.norm(w) + 1e-9)
    multi_modes = [m for m in modes if m not in binary_modes]
    for mode in multi_modes:
        X = data[f"X__{mode}__N{N_g}"][:, L_g, :]
        y = data[f"y__{mode}"]
        classes = np.unique(y)
        means = np.stack([X[y == c].mean(axis=0) for c in classes], axis=0)
        diffs = means - means.mean(axis=0, keepdims=True)
        pc = PCA(n_components=1).fit(diffs).components_[0]
        dirs[mode] = pc / (np.linalg.norm(pc) + 1e-9)
    keys = list(dirs.keys())
    cos = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            cos[i, j] = float(dirs[a] @ dirs[b])
    return pd.DataFrame(cos, index=keys, columns=keys)


def make_plots(scores, base_df, transfer_df, cos_df, modes, Ns):
    # 1) Heatmaps per mode
    for mode in modes:
        sub = scores[scores["mode"] == mode]
        if sub.empty:
            continue
        piv = sub.groupby(["N", "layer"])["acc"].mean().reset_index().pivot(
            index="N", columns="layer", values="acc"
        )
        plt.figure(figsize=(10, 3.5))
        sns.heatmap(piv, annot=True, fmt=".2f", cmap="viridis",
                    vmin=max(0.0, piv.values.min() - 0.05),
                    vmax=min(1.0, piv.values.max() + 0.05))
        plt.title(f"Probe accuracy — mode={mode}")
        plt.xlabel("Layer")
        plt.ylabel("Tokens N")
        plt.tight_layout()
        plt.savefig(FIGURES / f"heatmap_{mode}.png", dpi=120)
        plt.close()

    # 2) Inter-mode cosine
    plt.figure(figsize=(5, 4))
    sns.heatmap(cos_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Cosine similarity between mode directions")
    plt.tight_layout()
    plt.savefig(FIGURES / "mode_cosine.png", dpi=140)
    plt.close()

    # 3) Cross-transfer heatmap
    if not transfer_df.empty:
        tdf = transfer_df.pivot(
            index="train_mode", columns="test_mode", values="acc_mean"
        )
        plt.figure(figsize=(5, 4))
        sns.heatmap(tdf, annot=True, fmt=".2f", cmap="magma", vmin=0.5, vmax=1.0)
        plt.title("Cross-mode probe transfer (1D projection acc)")
        plt.tight_layout()
        plt.savefig(FIGURES / "cross_transfer.png", dpi=140)
        plt.close()

    # 4) Acc vs. layer (largest N)
    plt.figure(figsize=(8, 4.5))
    Nmax = max(Ns)
    for mode in modes:
        sub = scores[(scores["mode"] == mode) & (scores["N"] == Nmax)]
        if sub.empty:
            continue
        m = sub.groupby("layer")["acc"].mean()
        s = sub.groupby("layer")["acc"].std()
        plt.errorbar(m.index, m.values, yerr=s.values, label=mode, marker="o", capsize=2)
    plt.xlabel("Layer")
    plt.ylabel("Probe accuracy")
    plt.title(f"Probe accuracy vs. layer at N={Nmax}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "acc_vs_layer.png", dpi=140)
    plt.close()

    # 5) Probe vs. baseline bar chart
    best_neural = scores.groupby("mode")["acc"].max().reset_index()
    bar = base_df[["mode", "acc_mean"]].merge(best_neural, on="mode",
                                              suffixes=("_tfidf", "_neural_best"))
    bar = bar.rename(columns={"acc": "acc_neural_best", "acc_mean": "acc_tfidf"})
    bar = bar.melt(id_vars=["mode"], var_name="kind", value_name="accuracy")
    plt.figure(figsize=(7, 4))
    sns.barplot(data=bar, x="mode", y="accuracy", hue="kind")
    plt.title("Best neural probe vs. TF-IDF baseline")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIGURES / "probe_vs_baseline.png", dpi=140)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default=str(CACHE / "embeddings.npz"))
    parser.add_argument("--layer_stride", type=int, default=2)
    parser.add_argument("--modes", nargs="*", default=None,
                        help="Restrict to these modes for parallel running.")
    parser.add_argument("--output_suffix", default="",
                        help="Suffix for output filenames so parallel jobs don't clobber.")
    parser.add_argument("--skip_geometry", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_transfer", action="store_true")
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    all_modes = list(data["modes"])
    Ns = list(int(n) for n in data["Ns"])
    layers = list(int(L) for L in data["layers"])
    layers_used = layers[::args.layer_stride]
    modes = args.modes if args.modes else all_modes
    print(f"All modes={all_modes}  Running={modes}  Ns={Ns}  layers={layers}  using={layers_used}",
          file=sys.stderr, flush=True)

    # --- Main probe sweep ---
    scores = run_main_sweep(data, modes, Ns, layers_used)
    out_scores = RESULTS / f"probe_scores{args.output_suffix}.csv"
    scores.to_csv(out_scores, index=False)
    print(f"Wrote {out_scores}", file=sys.stderr, flush=True)

    if args.modes:
        # Subset run — don't compute global outputs (do those in a merge step).
        return

    # --- Baselines ---
    if not args.skip_baselines:
        print("\nRunning TF-IDF baselines...", file=sys.stderr, flush=True)
        base_df = run_tfidf_baselines(modes)
        base_df.to_csv(RESULTS / "baselines.csv", index=False)

    # --- Cross-mode transfer ---
    if not args.skip_transfer:
        print("\nRunning cross-mode transfer...", file=sys.stderr, flush=True)
        binary_modes = [m for m in modes if len(np.unique(data[f"y__{m}"])) == 2]
        transfer_df = run_cross_transfer(data, binary_modes, layers_used, Ns)
        transfer_df.to_csv(RESULTS / "cross_transfer.csv", index=False)
    else:
        binary_modes = [m for m in modes if len(np.unique(data[f"y__{m}"])) == 2]
        transfer_df = pd.DataFrame()

    # --- Geometry ---
    if not args.skip_geometry:
        print("\nComputing geometry...", file=sys.stderr, flush=True)
        geo_df = run_geometry(data, modes, layers_used, Ns)
        geo_df.to_csv(RESULTS / "geometry.csv", index=False)
        cos_df = run_inter_mode_cosine(data, modes, binary_modes, layers_used, Ns)
        cos_df.to_csv(RESULTS / "mode_direction_cosine.csv")
    else:
        cos_df = pd.DataFrame()

    # --- Plots ---
    base_df = pd.read_csv(RESULTS / "baselines.csv") if (RESULTS / "baselines.csv").exists() else pd.DataFrame()
    if not base_df.empty:
        make_plots(scores, base_df, transfer_df, cos_df, modes, Ns)

    print("\nDone.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
