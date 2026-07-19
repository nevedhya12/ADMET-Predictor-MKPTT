"""
full_descriptor_audit.py

Replaces manual/curated descriptor selection (NOISE_TO_DROP in
master_scaffold_pipeline.py) with a fully data-driven process:

  1. Compute the ENTIRE RDKit 2D descriptor catalogue (~200 descriptors,
     via Descriptors.descList) for every molecule per category -- not just
     the 15 hand-picked ones.
  2. Automatically drop descriptors that are degenerate (all-NaN or
     zero-variance across the dataset) -- these carry no information by
     construction, not by judgment call.
  3. Concatenate with the fingerprint embedding (radius=3, UMAP n=10 --
     already validated in prior ablations) and train XGBoost across the
     same scaffold-CV folds used everywhere else in this project.
  4. Average feature importance (gain-based) for every descriptor ACROSS
     ALL 5 FOLDS, not a single fit -- avoids one lucky/unlucky fold
     deciding which descriptors "matter."
  5. Rank every descriptor by its measured importance share and keep only
     those clearing IMPORTANCE_THRESHOLD (as a fraction of total model
     importance) -- purely a number, not a guess.
  6. Re-run scaffold CV with the trimmed descriptor set and report the
     final metric, compared directly against the current curated
     (NOISE_TO_DROP-based) result, so you can see whether math-driven
     selection actually matches or beats manual curation.

Output per category:
  - scaffold_dataset/{category}_descriptor_importance.csv  (every descriptor
    ranked by importance, full audit trail)
  - scaffold_dataset/descriptor_keep_lists.json  (final math-driven keep
    list per category -- drop this straight into your pipeline)
  - printed comparison: full-descriptor-set score vs trimmed-set score vs
    your existing curated-set score
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import umap
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, r2_score, mean_squared_error,
)
import xgboost as xgb
from tdc.single_pred import ADME, Tox

warnings.filterwarnings("ignore")

OUTPUT_DIR = "scaffold_dataset"
RANDOM_STATE = 42
N_FOLDS = 5
FP_RADIUS = 3           # validated via prior ablation
FP_N_COMPONENTS = 10    # validated via prior ablation

SKEW_THRESHOLD = 1.0
CEILING_FRACTION_THRESHOLD = 0.03
CEILING_ATOL = 1e-6

# A descriptor must account for at least this fraction of TOTAL model
# importance (fingerprint dims + all descriptors combined) to be kept.
# Purely a number -- not tuned per category, applied identically everywhere.
IMPORTANCE_THRESHOLD = 0.001  # 0.1%

ENDPOINTS = {
    "Absorption":   {"class": ADME, "name": "Caco2_Wang"},
    "Distribution": {"class": ADME, "name": "BBB_Martins"},
    "Metabolism":   {"class": ADME, "name": "CYP2C9_Veith"},
    "Excretion":    {"class": ADME, "name": "Clearance_Hepatocyte_AZ"},
    "Toxicity":     {"class": Tox,  "name": "hERG"},
}

# The current curated set (from master_scaffold_pipeline.py) -- kept here
# ONLY so we can print a fair comparison at the end.
CURRENT_CURATED_DROP = {
    "Absorption": ["FormalCharge", "RotBonds", "HBA", "QED", "BertzCT", "AromaticRings"],
    "Distribution": ["FormalCharge"],
    "Metabolism": [],
    "Excretion": ["FormalCharge"],
    "Toxicity": []
}
CURATED_DESCRIPTOR_NAMES = [
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
    "AromaticRings", "FormalCharge", "MolarRefractivity", "Fsp3", "QED",
    "AmideBondCount", "MaxRingSize", "BertzCT", "StereocenterCount",
]

# Full RDKit descriptor catalogue -- name + function pairs
ALL_RDKIT_DESCRIPTORS = Descriptors.descList  # list of (name, function)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def compute_all_descriptors(mol):
    """Computes every descriptor in RDKit's catalogue for one molecule.
    Individual failures become NaN rather than aborting the whole row."""
    values = np.empty(len(ALL_RDKIT_DESCRIPTORS), dtype=np.float64)
    for i, (name, func) in enumerate(ALL_RDKIT_DESCRIPTORS):
        try:
            v = func(mol)
            values[i] = v if np.isfinite(v) else np.nan
        except Exception:
            values[i] = np.nan
    return values


def create_scaffold_folds(mols, n_folds=5):
    scaffolds = {}
    for idx, mol in enumerate(mols):
        if mol is None:
            continue
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(core)
        except:
            scaffold_smiles = ""
        scaffolds.setdefault(scaffold_smiles, []).append(idx)

    sorted_scaffolds = sorted(scaffolds.values(), key=len, reverse=True)
    fold_assignments = {i: [] for i in range(n_folds)}
    fold_sizes = np.zeros(n_folds)
    for scaffold_indices in sorted_scaffolds:
        smallest_fold = np.argmin(fold_sizes)
        fold_assignments[smallest_fold].extend(scaffold_indices)
        fold_sizes[smallest_fold] += len(scaffold_indices)

    clusters = np.zeros(len(mols), dtype=int)
    for fold_id, indices in fold_assignments.items():
        for idx in indices:
            clusters[idx] = fold_id
    return clusters


def decide_log_transform(y):
    skew = pd.Series(y).skew()
    if skew > SKEW_THRESHOLD and np.all(y >= 0):
        return np.log1p(y), True, skew
    return y, False, skew


def detect_ceiling(y):
    y_max = y.max()
    frac = np.mean(np.isclose(y, y_max, atol=CEILING_ATOL))
    return frac >= CEILING_FRACTION_THRESHOLD, y_max, frac


def run_scaffold_cv_with_importance(X, y, y_original, scaffold_folds, is_classification,
                                     log_transformed, has_ceiling, ceiling_value, n_fp_dims):
    """Runs scaffold CV, returns (avg_metric_a, avg_metric_b, avg_feature_importance)."""
    fold_metric_a, fold_metric_b, fold_weights = [], [], []
    importances = []

    for test_fold in range(N_FOLDS):
        test_mask = (scaffold_folds == test_fold)
        train_mask = (scaffold_folds != test_fold)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        y_test_original = y_original[test_mask]

        if len(y_test) == 0:
            continue

        if is_classification:
            if len(np.unique(y_test)) < 2:
                continue
            model = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            fold_metric_a.append(roc_auc_score(y_test, probs))
            fold_metric_b.append(average_precision_score(y_test, probs))
            fold_weights.append(len(y_test))
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=RANDOM_STATE, eval_metric='rmse', n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            if log_transformed:
                preds = np.expm1(preds)
            if has_ceiling:
                preds = np.clip(preds, None, ceiling_value)
            fold_metric_a.append(r2_score(y_test_original, preds))
            fold_metric_b.append(np.sqrt(mean_squared_error(y_test_original, preds)))
            fold_weights.append(len(y_test))

        importances.append(model.feature_importances_)

    if len(fold_metric_a) == 0:
        return None, None, None

    avg_importance = np.mean(importances, axis=0)
    return (
        np.average(fold_metric_a, weights=fold_weights),
        np.average(fold_metric_b, weights=fold_weights),
        avg_importance,
    )


def process_category(category, details):
    print(f"\n{'=' * 60}")
    print(f"Category: {category}")
    print(f"{'=' * 60}")

    raw_df = details["class"](name=details["name"]).get_data()
    mols = [Chem.MolFromSmiles(s) for s in raw_df['Drug']]
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in valid_idx]
    y = raw_df['Y'].values[valid_idx].astype(np.float64)

    is_classification = len(np.unique(y)) <= 10
    metric_labels = ("ROC-AUC", "AUPRC") if is_classification else ("R2", "RMSE")

    y_original = y.copy()
    log_transformed, has_ceiling, ceiling_value = False, False, None
    if not is_classification:
        y, log_transformed, skew = decide_log_transform(y)
        has_ceiling, ceiling_value, ceiling_frac = detect_ceiling(y_original)

    # --- Full descriptor sweep ---
    print(f" -> Computing all {len(ALL_RDKIT_DESCRIPTORS)} RDKit descriptors for {len(mols)} molecules...")
    desc_full = np.stack([compute_all_descriptors(m) for m in mols])
    desc_names_full = [name for name, _ in ALL_RDKIT_DESCRIPTORS]

    # Drop degenerate columns purely on math: all-NaN or zero-variance
    nan_frac = np.mean(np.isnan(desc_full), axis=0)
    with np.errstate(invalid="ignore"):
        col_std = np.nanstd(desc_full, axis=0)
    degenerate_mask = (nan_frac >= 0.99) | (col_std == 0) | np.isnan(col_std)
    n_degenerate = degenerate_mask.sum()
    print(f" -> Dropping {n_degenerate} degenerate descriptors (all-NaN or zero-variance)")

    desc_full = desc_full[:, ~degenerate_mask]
    desc_names_full = [n for n, keep in zip(desc_names_full, ~degenerate_mask) if keep]

    # Impute remaining NaNs with column median (math-driven, not per-molecule guessing)
    col_medians = np.nanmedian(desc_full, axis=0)
    nan_idx = np.where(np.isnan(desc_full))
    desc_full[nan_idx] = np.take(col_medians, nan_idx[1])

    desc_scaled = StandardScaler().fit_transform(desc_full)

    # --- Fingerprint embedding (fixed, already validated) ---
    print(" -> Building fingerprint embedding...")
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=2048)
    fp_list = []
    for mol in mols:
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((0,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr)
    X_fp = np.stack(fp_list)
    reducer = umap.UMAP(n_components=FP_N_COMPONENTS, metric="jaccard", random_state=RANDOM_STATE)
    fp_embedding = reducer.fit_transform(X_fp)

    X_full = np.hstack([fp_embedding, desc_scaled])
    scaffold_folds = create_scaffold_folds(mols, n_folds=N_FOLDS)

    # --- Run CV with ALL descriptors, get importance ---
    print(" -> Training with FULL descriptor set to measure importance...")
    metric_a_full, metric_b_full, importance = run_scaffold_cv_with_importance(
        X_full, y, y_original, scaffold_folds, is_classification,
        log_transformed, has_ceiling, ceiling_value, FP_N_COMPONENTS
    )

    if importance is None:
        print(f" -> WARNING: no valid folds for {category}, skipping")
        return None, None

    desc_importance = importance[FP_N_COMPONENTS:]
    total_importance = importance.sum()
    importance_share = desc_importance / total_importance if total_importance > 0 else desc_importance

    audit_df = pd.DataFrame({
        "descriptor": desc_names_full,
        "importance_share": importance_share,
    }).sort_values("importance_share", ascending=False)

    audit_path = f"{OUTPUT_DIR}/{category}_descriptor_importance.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f" -> Saved full importance ranking -> {audit_path}")

    keep_mask = importance_share >= IMPORTANCE_THRESHOLD
    keep_list = [n for n, k in zip(desc_names_full, keep_mask) if k]
    print(f" -> {len(keep_list)}/{len(desc_names_full)} descriptors clear the "
          f"{IMPORTANCE_THRESHOLD*100:.1f}% importance threshold")
    print(f" -> Top 10 by importance: {audit_df.head(10)['descriptor'].tolist()}")

    # --- Re-run CV with ONLY the math-selected descriptors ---
    keep_indices_desc = [i for i, k in enumerate(keep_mask) if k]
    desc_scaled_trimmed = desc_scaled[:, keep_indices_desc]
    X_trimmed = np.hstack([fp_embedding, desc_scaled_trimmed])

    metric_a_trim, metric_b_trim, _ = run_scaffold_cv_with_importance(
        X_trimmed, y, y_original, scaffold_folds, is_classification,
        log_transformed, has_ceiling, ceiling_value, FP_N_COMPONENTS
    )

    # --- For comparison: re-run with your EXISTING curated set ---
    curated_drop = CURRENT_CURATED_DROP.get(category, [])
    curated_keep_names = [n for n in CURATED_DESCRIPTOR_NAMES if n not in curated_drop]
    # Only descriptors that exist in both the curated 15-set AND full RDKit catalogue
    curated_indices = [desc_names_full.index(n) for n in curated_keep_names if n in desc_names_full]
    if curated_indices:
        desc_scaled_curated = desc_scaled[:, curated_indices]
        X_curated = np.hstack([fp_embedding, desc_scaled_curated])
        metric_a_cur, metric_b_cur, _ = run_scaffold_cv_with_importance(
            X_curated, y, y_original, scaffold_folds, is_classification,
            log_transformed, has_ceiling, ceiling_value, FP_N_COMPONENTS
        )
    else:
        metric_a_cur, metric_b_cur = None, None

    print(f"\n -> {metric_labels[0]} comparison for {category}:")
    print(f"      Full RDKit set ({len(desc_names_full)} descriptors):    {metric_a_full:.4f}")
    print(f"      Math-trimmed set ({len(keep_list)} descriptors):        {metric_a_trim:.4f}")
    if metric_a_cur is not None:
        print(f"      Your curated set ({len(curated_keep_names)} descriptors):       {metric_a_cur:.4f}")

    result_row = {
        "category": category,
        "metric_label": metric_labels[0],
        "full_set_score": metric_a_full,
        "trimmed_set_score": metric_a_trim,
        "curated_set_score": metric_a_cur,
        "n_full": len(desc_names_full),
        "n_trimmed": len(keep_list),
        "n_curated": len(curated_keep_names),
    }

    return keep_list, result_row


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== FULL RDKIT DESCRIPTOR AUDIT ===")
    print(f" -> Testing all {len(ALL_RDKIT_DESCRIPTORS)} RDKit 2D descriptors per category")
    print(f" -> Importance threshold: {IMPORTANCE_THRESHOLD*100:.1f}% of total model importance")

    keep_lists = {}
    comparison_rows = []

    for category, details in ENDPOINTS.items():
        keep_list, result_row = process_category(category, details)
        if keep_list is not None:
            keep_lists[category] = keep_list
            comparison_rows.append(result_row)

    keep_lists_path = f"{OUTPUT_DIR}/descriptor_keep_lists.json"
    with open(keep_lists_path, "w") as f:
        json.dump(keep_lists, f, indent=2)
    print(f"\nSaved final math-driven keep lists -> {keep_lists_path}")

    print(f"\n{'=' * 60}")
    print("SUMMARY: FULL vs MATH-TRIMMED vs YOUR CURATED SET")
    print(f"{'=' * 60}")
    comparison_df = pd.DataFrame(comparison_rows)
    print(comparison_df.to_string(index=False))

    print("""
Next step: replace NOISE_TO_DROP in master_scaffold_pipeline.py with the
keep lists in descriptor_keep_lists.json (invert: whatever is NOT in the
keep list for a category gets dropped for that category), then re-run the
final pipeline to get your official numbers with the math-driven descriptor
set.
""")