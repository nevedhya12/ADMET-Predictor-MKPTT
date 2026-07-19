"""
optimize_fingerprint_radius.py

Ablates Morgan fingerprint radius (currently hardcoded to 3 in
master_scaffold_pipeline.py) instead of assuming it. Same rationale as the
earlier UMAP-dimensionality and cluster-count ablations: don't guess a
hyperparameter, measure it.

For each candidate radius, this rebuilds the ENTIRE feature pipeline exactly
as master_scaffold_pipeline.py does (fingerprint -> UMAP -> concat with
descriptors -> scaffold split -> XGBoost) and reports the SAME required
metrics (ROC-AUC/AUPRC for classification, R2/RMSE for regression).

Important: the Murcko scaffold assignment itself does NOT depend on radius
(scaffolds are computed directly from molecular structure, not from the
fingerprint), so scaffold folds are computed ONCE per category and reused
across every radius -- this isolates radius as the only thing changing
between runs, the same way the multi-seed cluster ablation isolated
K-means initialization.

Output:
  - scaffold_dataset/fingerprint_radius_ablation_results.csv
  - scaffold_dataset/fingerprint_radius_ablation_summary.png
  - printed recommendation per category
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, GraphDescriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, r2_score, mean_squared_error,
)
import xgboost as xgb
from tdc.single_pred import ADME, Tox

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG -- kept identical to master_scaffold_pipeline.py wherever it
# doesn't concern radius, so results are directly comparable to your
# existing final numbers.
# ---------------------------------------------------------------------------
OUTPUT_DIR = "scaffold_dataset"
RANDOM_STATE = 42
N_FOLDS = 5
FP_N_COMPONENTS = 10
CANDIDATE_RADII = [1, 2, 3, 4, 5]

SKEW_THRESHOLD = 1.0
CEILING_FRACTION_THRESHOLD = 0.03
CEILING_ATOL = 1e-6

ENDPOINTS = {
    "Absorption":   {"class": ADME, "name": "Caco2_Wang"},
    "Distribution": {"class": ADME, "name": "BBB_Martins"},
    "Metabolism":   {"class": ADME, "name": "CYP2C9_Veith"},
    "Excretion":    {"class": ADME, "name": "Clearance_Hepatocyte_AZ"},
    "Toxicity":     {"class": Tox,  "name": "hERG"},
}

DESCRIPTOR_NAMES = [
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
    "AromaticRings", "FormalCharge", "MolarRefractivity", "Fsp3", "QED",
    "AmideBondCount", "MaxRingSize", "BertzCT", "StereocenterCount",
]

NOISE_TO_DROP = {
    "Absorption": ["FormalCharge", "RotBonds", "HBA", "QED", "BertzCT", "AromaticRings"],
    "Distribution": ["FormalCharge"],
    "Metabolism": [],
    "Excretion": ["FormalCharge"],
    "Toxicity": []
}

_AMIDE_PATTERN = Chem.MolFromSmarts("C(=O)N")


# ---------------------------------------------------------------------------
# HELPERS -- identical to master_scaffold_pipeline.py
# ---------------------------------------------------------------------------
def get_2d_descriptors(mol):
    if mol is None:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float64)
    try:
        amide_count = len(mol.GetSubstructMatches(_AMIDE_PATTERN))
        ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
        max_ring_size = max(ring_sizes) if ring_sizes else 0
        stereo_count = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
        return np.array([
            Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol), Chem.GetFormalCharge(mol), Crippen.MolMR(mol),
            rdMolDescriptors.CalcFractionCSP3(mol), Descriptors.qed(mol), amide_count,
            max_ring_size, GraphDescriptors.BertzCT(mol), stereo_count,
        ], dtype=np.float64)
    except:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float64)


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
    frac_at_ceiling = np.mean(np.isclose(y, y_max, atol=CEILING_ATOL))
    has_ceiling = frac_at_ceiling >= CEILING_FRACTION_THRESHOLD
    return has_ceiling, y_max, frac_at_ceiling


def run_scaffold_cv(X, y, y_original, scaffold_folds, is_classification, log_transformed, has_ceiling, ceiling_value):
    fold_metric_a, fold_metric_b, fold_weights = [], [], []

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

    if len(fold_metric_a) == 0:
        return None, None

    return (
        np.average(fold_metric_a, weights=fold_weights),
        np.average(fold_metric_b, weights=fold_weights),
    )


def ablate_category(category, details):
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

    # Descriptors don't depend on radius -- compute once, reuse across radii
    desc_matrix = np.stack([get_2d_descriptors(m) for m in mols])
    drop_list = NOISE_TO_DROP.get(category, [])
    if drop_list:
        keep_indices = [i for i, name in enumerate(DESCRIPTOR_NAMES) if name not in drop_list]
        desc_matrix = desc_matrix[:, keep_indices]
    desc_scaled = StandardScaler().fit_transform(desc_matrix)

    # Scaffold assignment doesn't depend on radius either -- compute once,
    # so every radius is evaluated on the EXACT SAME folds (fair comparison)
    scaffold_folds = create_scaffold_folds(mols, n_folds=N_FOLDS)

    results = []
    for radius in CANDIDATE_RADII:
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2048)
        fp_list = []
        for mol in mols:
            fp = morgan_gen.GetFingerprint(mol)
            arr = np.zeros((0,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fp_list.append(arr)
        X_fp = np.stack(fp_list)

        reducer = umap.UMAP(n_components=FP_N_COMPONENTS, metric="jaccard", random_state=RANDOM_STATE)
        fp_embedding = reducer.fit_transform(X_fp)

        X_combined = np.hstack([fp_embedding, desc_scaled])

        metric_a, metric_b = run_scaffold_cv(
            X_combined, y, y_original, scaffold_folds,
            is_classification, log_transformed, has_ceiling, ceiling_value,
        )

        if metric_a is None:
            print(f" -> radius={radius}: no valid folds, skipped")
            continue

        print(f" -> radius={radius} | {metric_labels[0]}={metric_a:.4f} | {metric_labels[1]}={metric_b:.4f}")

        results.append({
            "category": category, "radius": radius,
            "metric_a": metric_a, "metric_b": metric_b,
            "metric_a_label": metric_labels[0], "metric_b_label": metric_labels[1],
        })

    return pd.DataFrame(results)


def recommend_best(df):
    recommendations = {}
    for category, group in df.groupby("category"):
        best_row = group.loc[group["metric_a"].idxmax()]
        recommendations[category] = {
            "radius": int(best_row["radius"]),
            "metric_a": best_row["metric_a"],
            "metric_a_label": best_row["metric_a_label"],
        }
    return recommendations


def plot_results(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for category, group in df.groupby("category"):
        group = group.sort_values("radius")
        label = f"{category} ({group['metric_a_label'].iloc[0]})"
        ax.plot(group["radius"], group["metric_a"], marker="o", label=label)

    ax.set_title("Primary metric vs Morgan fingerprint radius")
    ax.set_xlabel("Radius")
    ax.set_ylabel("Metric value (ROC-AUC or R2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved comparison plot -> {out_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    for category, details in ENDPOINTS.items():
        df_cat = ablate_category(category, details)
        all_results.append(df_cat)

    full_df = pd.concat(all_results, ignore_index=True)

    csv_path = f"{OUTPUT_DIR}/fingerprint_radius_ablation_results.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results -> {csv_path}")

    plot_results(full_df, f"{OUTPUT_DIR}/fingerprint_radius_ablation_summary.png")

    recommendations = recommend_best(full_df)

    print(f"\n{'=' * 60}")
    print("RECOMMENDED RADIUS PER CATEGORY")
    print(f"{'=' * 60}")
    for category, rec in recommendations.items():
        print(f" -> {category:<15s}: radius = {rec['radius']}  "
              f"({rec['metric_a_label']} = {rec['metric_a']:.4f})")

    print("""
Note: scaffold folds and descriptors are held IDENTICAL across every radius
tested per category, so radius is isolated as the only variable -- this
comparison is fair in a way the earlier "radius didn't seem to matter" check
(done informally, on the old K-means pipeline) was not.

If gains are small/flat across radii, that's a legitimate finding worth
keeping in your README: it means the fingerprint's radius is not a
meaningful bottleneck for this dataset, and your FP_RADIUS choice can be
justified by this ablation rather than by assumption.
""")