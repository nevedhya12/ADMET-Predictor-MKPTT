"""
optimize_umap_dims.py

Goal: find the best n_components for compressing the 2048-bit fingerprint
via UMAP, instead of guessing (your current code hardcodes n_components=3).

Method:
  For each candidate dimensionality D in CANDIDATE_DIMS:
    1. Fit UMAP(n_components=D, metric='jaccard') on the raw fingerprint
    2. Measure trustworthiness (how well local structure from the original
       2048-dim space survives compression to D dims) -- this is independent
       of any downstream model, it just measures information loss.
    3. Re-run KMeans(n_clusters=5) on the D-dim embedding to get fresh
       spatial folds
    4. Run the exact same spatial cross-validation you're already using
       (train on 4 cluster-folds, test on the 5th, weighted average by
       fold size) with XGBoost, to get a real downstream performance number
  This gives you two independent signals (trustworthiness + weighted CV
  score) per dimensionality, per ADMET category, so you can pick the
  dimensionality that stops giving meaningful returns (the "elbow").

Output:
  - processed_data/umap_dim_ablation_results.csv (raw results, every row is
    one category x dimensionality combination)
  - processed_data/umap_dim_ablation_summary.png (weighted score & trustworthiness
    vs n_components, one line per category)
  - printed recommendation per category and an overall recommendation

Note: descriptors are NOT included yet, per your request -- this script only
optimizes the fingerprint compression. Once you've picked a dimensionality,
plug that into your existing clustering / training scripts.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ALL_CATEGORIES = ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]

# Dimensionalities to test. Kept modest at the high end since fingerprints
# are sparse/binary and returns typically flatten out well under 100 dims.
CANDIDATE_DIMS = [3, 5, 10, 20, 30, 50, 75, 100]

N_CLUSTERS = 5
RANDOM_STATE = 42
DATA_DIR = "processed_data"

# Trustworthiness is O(n^2 log n) — cap the sample size used for it so large
# datasets (e.g. Metabolism with 6500+ rows) don't stall the script.
TRUSTWORTHINESS_MAX_SAMPLES = 2000


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def compute_trustworthiness(X_high, X_low, n_neighbors=15, max_samples=TRUSTWORTHINESS_MAX_SAMPLES):
    """
    Trustworthiness in [0, 1]: how much the local neighborhood structure of
    the original high-dim space is preserved in the low-dim embedding.
    1.0 = perfect preservation, closer to 0 = badly distorted.

    Subsamples for speed on large datasets -- this is only a diagnostic,
    not something we need exact on the full dataset.
    """
    n = X_high.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]
    return trustworthiness(X_high, X_low, n_neighbors=n_neighbors, metric="jaccard")


def run_spatial_cv(X_embedded, y, cluster_groups):
    """
    Same spatial cross-validation logic as your train_xgboost.py:
    for each cluster, train on the rest, test on that cluster, weight the
    average by fold size. Returns (weighted_score, task_type).
    """
    unique_clusters = np.unique(cluster_groups)
    is_classification = len(np.unique(y)) <= 10

    fold_scores = []
    fold_weights = []

    for test_cluster in unique_clusters:
        test_mask = cluster_groups == test_cluster
        train_mask = ~test_mask

        X_train, X_test = X_embedded[train_mask], X_embedded[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(y_test) == 0 or len(y_train) == 0:
            continue

        if is_classification:
            model = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=RANDOM_STATE, eval_metric="rmse", n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)

        fold_scores.append(score)
        fold_weights.append(len(y_test))

    weighted_score = np.average(fold_scores, weights=fold_weights)
    task_type = "Classification" if is_classification else "Regression"
    return weighted_score, task_type


def ablate_category(category):
    print(f"\n{'=' * 60}")
    print(f"Category: {category}")
    print(f"{'=' * 60}")

    X = np.load(f"{DATA_DIR}/{category}_X.npy")
    y = np.load(f"{DATA_DIR}/{category}_y.npy")
    print(f" -> Loaded {X.shape[0]} molecules, {X.shape[1]} fingerprint bits")

    results = []

    for dims in CANDIDATE_DIMS:
        t0 = time.time()

        reducer = umap.UMAP(n_components=dims, metric="jaccard", random_state=RANDOM_STATE)
        X_embedded = reducer.fit_transform(X)

        tw = compute_trustworthiness(X, X_embedded)

        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        cluster_groups = kmeans.fit_predict(X_embedded)

        weighted_score, task_type = run_spatial_cv(X_embedded, y, cluster_groups)

        elapsed = time.time() - t0

        print(
            f" -> n_components={dims:>3d} | trustworthiness={tw:.4f} | "
            f"weighted {('accuracy' if task_type == 'Classification' else 'R2'):>8s}={weighted_score:.4f} | "
            f"{elapsed:.1f}s"
        )

        results.append({
            "category": category,
            "n_components": dims,
            "trustworthiness": tw,
            "weighted_score": weighted_score,
            "task_type": task_type,
            "seconds": elapsed,
        })

    return pd.DataFrame(results)


def recommend_best(df):
    """
    Pick the dimensionality per category as the smallest D where weighted_score
    is within 1% (absolute) of the max score achieved across all D -- i.e. the
    elbow point, not just whichever D happened to score highest (higher D can
    win by noise, not by real signal).
    """
    recommendations = {}
    for category, group in df.groupby("category"):
        best_score = group["weighted_score"].max()
        threshold = best_score - 0.01
        candidates = group[group["weighted_score"] >= threshold].sort_values("n_components")
        chosen = candidates.iloc[0]
        recommendations[category] = int(chosen["n_components"])
    return recommendations


def plot_results(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for category, group in df.groupby("category"):
        group = group.sort_values("n_components")
        axes[0].plot(group["n_components"], group["weighted_score"], marker="o", label=category)
        axes[1].plot(group["n_components"], group["trustworthiness"], marker="o", label=category)

    axes[0].set_title("Weighted downstream CV score vs UMAP n_components")
    axes[0].set_xlabel("n_components")
    axes[0].set_ylabel("Weighted score (accuracy or R2)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Trustworthiness vs UMAP n_components")
    axes[1].set_xlabel("n_components")
    axes[1].set_ylabel("Trustworthiness (0-1)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved comparison plot -> {out_path}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    all_results = []
    for category in ALL_CATEGORIES:
        df_cat = ablate_category(category)
        all_results.append(df_cat)

    full_df = pd.concat(all_results, ignore_index=True)

    csv_path = f"{DATA_DIR}/umap_dim_ablation_results.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results -> {csv_path}")

    plot_results(full_df, f"{DATA_DIR}/umap_dim_ablation_summary.png")

    recommendations = recommend_best(full_df)

    print(f"\n{'=' * 60}")
    print("RECOMMENDED n_components PER CATEGORY (elbow point)")
    print(f"{'=' * 60}")
    for category, dims in recommendations.items():
        print(f" -> {category:<15s}: {dims} dimensions")

    overall = int(round(np.median(list(recommendations.values()))))
    print(f"\nIf you want ONE shared n_components across all 5 categories")
    print(f"(simplest / most consistent pipeline), use: {overall}")
    print("\nNote: this only compresses the fingerprint. Add descriptors")
    print("as a separate concatenated block afterward, per Option 1.")