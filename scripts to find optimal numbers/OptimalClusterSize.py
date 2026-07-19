"""
optimize_cluster_count_multiseed.py

Same ablation as optimize_cluster_count.py, but repeats each (category, K)
combination across multiple KMeans random seeds and reports the MEAN and
STANDARD DEVIATION of both the weighted score and the min fold size.

Why: a single random_state=42 run could get a lucky or unlucky cluster
split, especially for categories with imbalanced chemical space (Toxicity,
Excretion). If a K's score is high on one seed but the std across seeds is
large, that's a fluke, not a real effect -- this script makes that visible
instead of hiding it in one number.

Output:
  - processed_data/cluster_count_multiseed_results.csv (every category x K x seed row)
  - processed_data/cluster_count_multiseed_summary.csv (aggregated mean/std per category x K)
  - processed_data/cluster_count_multiseed_summary.png (mean score with error bars)
  - printed recommendation per category, now justified by seed-stability, not a single run
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score

warnings.filterwarnings("ignore")

DATA_DIR = "processed_data"
ALL_CATEGORIES = ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]

CANDIDATE_K = [3, 4, 5, 6, 7, 8, 10]

# Multiple KMeans random states -- this is the whole point of this script.
# XGBoost's random_state is kept fixed (42) across all runs so the ONLY
# thing varying between runs is the cluster assignment, isolating that as
# the source of any variance we see.
SEEDS = [0, 7, 42, 123, 2024]

XGB_RANDOM_STATE = 42
SMALL_FOLD_THRESHOLD = 30

# A K's mean score must be within this margin of the best mean score to be
# considered "not meaningfully different" when picking a winner -- avoids
# picking a K that only wins by noise-level margins
SCORE_EQUIVALENCE_MARGIN = 0.02


def run_spatial_cv(X, y, cluster_groups):
    unique_clusters = np.unique(cluster_groups)
    is_classification = len(np.unique(y)) <= 10

    fold_scores = []
    fold_weights = []

    for test_cluster in unique_clusters:
        test_mask = cluster_groups == test_cluster
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(y_test) == 0 or len(y_train) == 0:
            continue

        if is_classification:
            model = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=XGB_RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=XGB_RANDOM_STATE, eval_metric="rmse", n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)

        fold_scores.append(score)
        fold_weights.append(len(y_test))

    weighted_score = np.average(fold_scores, weights=fold_weights)
    task_type = "Classification" if is_classification else "Regression"
    min_fold_size = int(min(fold_weights))
    return weighted_score, task_type, min_fold_size


def ablate_category(category):
    print(f"\n{'=' * 60}")
    print(f"Category: {category}")
    print(f"{'=' * 60}")

    X = np.load(f"{DATA_DIR}/{category}_combined_X.npy")
    y = np.load(f"{DATA_DIR}/{category}_y.npy")
    print(f" -> Loaded combined features: {X.shape}  |  seeds per K: {len(SEEDS)}")

    raw_results = []

    for k in CANDIDATE_K:
        seed_scores = []
        seed_min_folds = []

        for seed in SEEDS:
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            cluster_groups = kmeans.fit_predict(X)
            weighted_score, task_type, min_fold_size = run_spatial_cv(X, y, cluster_groups)

            seed_scores.append(weighted_score)
            seed_min_folds.append(min_fold_size)

            raw_results.append({
                "category": category, "K": k, "seed": seed,
                "weighted_score": weighted_score, "task_type": task_type,
                "min_fold_size": min_fold_size,
            })

        mean_score = np.mean(seed_scores)
        std_score = np.std(seed_scores)
        mean_min_fold = np.mean(seed_min_folds)

        metric_label = "accuracy" if task_type == "Classification" else "R2"
        flag = " <- HIGH VARIANCE ACROSS SEEDS" if std_score > 0.05 else ""
        print(
            f" -> K={k:>2d} | mean {metric_label}={mean_score:.4f} (+/- {std_score:.4f}) | "
            f"mean min_fold_size={mean_min_fold:.0f}{flag}"
        )

    return pd.DataFrame(raw_results)


def summarize(df):
    summary = df.groupby(["category", "K"]).agg(
        mean_score=("weighted_score", "mean"),
        std_score=("weighted_score", "std"),
        mean_min_fold=("min_fold_size", "mean"),
        min_min_fold=("min_fold_size", "min"),
        task_type=("task_type", "first"),
    ).reset_index()
    return summary


def recommend_best(summary):
    """
    Per category: among K values whose mean_min_fold clears the stability
    threshold, find the best mean_score, then prefer the SMALLEST K whose
    mean_score is within SCORE_EQUIVALENCE_MARGIN of that best -- avoids
    picking a more complex (higher K) split when a simpler one performs
    statistically indistinguishably.
    """
    recommendations = {}
    for category, group in summary.groupby("category"):
        stable = group[group["mean_min_fold"] >= SMALL_FOLD_THRESHOLD]
        pool = stable if len(stable) > 0 else group
        forced = len(stable) == 0

        best_score = pool["mean_score"].max()
        near_best = pool[pool["mean_score"] >= best_score - SCORE_EQUIVALENCE_MARGIN]
        chosen = near_best.sort_values("K").iloc[0]

        recommendations[category] = {
            "K": int(chosen["K"]),
            "mean_score": chosen["mean_score"],
            "std_score": chosen["std_score"],
            "forced": forced,
        }
    return recommendations


def plot_results(summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for category, group in summary.groupby("category"):
        group = group.sort_values("K")
        axes[0].errorbar(group["K"], group["mean_score"], yerr=group["std_score"],
                          marker="o", capsize=3, label=category)
        axes[1].plot(group["K"], group["mean_min_fold"], marker="o", label=category)

    axes[0].set_title(f"Mean weighted CV score vs K ({len(SEEDS)} seeds, error bars = std)")
    axes[0].set_xlabel("K (number of clusters)")
    axes[0].set_ylabel("Mean weighted score")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(SMALL_FOLD_THRESHOLD, color="red", linestyle="--", linewidth=1, alpha=0.6,
                     label=f"stability threshold ({SMALL_FOLD_THRESHOLD})")
    axes[1].set_title("Mean smallest fold size vs K")
    axes[1].set_xlabel("K (number of clusters)")
    axes[1].set_ylabel("Mean min fold size")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved comparison plot -> {out_path}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    all_raw = []
    for category in ALL_CATEGORIES:
        df_cat = ablate_category(category)
        all_raw.append(df_cat)

    raw_df = pd.concat(all_raw, ignore_index=True)
    raw_path = f"{DATA_DIR}/cluster_count_multiseed_results.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved raw per-seed results -> {raw_path}")

    summary_df = summarize(raw_df)
    summary_path = f"{DATA_DIR}/cluster_count_multiseed_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved aggregated summary -> {summary_path}")

    plot_results(summary_df, f"{DATA_DIR}/cluster_count_multiseed_summary.png")

    recommendations = recommend_best(summary_df)

    print(f"\n{'=' * 60}")
    print(f"RECOMMENDED K PER CATEGORY (averaged over {len(SEEDS)} seeds: {SEEDS})")
    print(f"{'=' * 60}")
    for category, rec in recommendations.items():
        note = "  (no K fully clears stability threshold across seeds -- best available)" if rec["forced"] else ""
        print(f" -> {category:<15s}: K = {rec['K']}  "
              f"(mean score = {rec['mean_score']:.4f} +/- {rec['std_score']:.4f}){note}")

    print("""
This recommendation is now defensible against the "was it just luck"
question: each K's score is an average across 5 different random cluster
initializations, and the reported std tells you how much that score
actually moves with the seed. A small std means the K genuinely works
well regardless of initialization -- not because one lucky split happened.

For your final pipeline, you can either:
  (a) pick ONE seed (e.g. 42) and the recommended K, and note in your
      writeup that you validated it's stable across 5 seeds, or
  (b) go further and ensemble: train on all 5 seed-based cluster
      assignments and average the resulting fold scores as your final
      reported number (more defensible, more compute).
""")