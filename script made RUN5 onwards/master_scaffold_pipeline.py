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

# 1. Clean Terminal Output
warnings.filterwarnings("ignore")

# 2. Configuration
OUTPUT_DIR = "scaffold_dataset"
RANDOM_STATE = 42
N_FOLDS = 5
FP_RADIUS = 3          # Using the expanded structural vocabulary
FP_N_COMPONENTS = 10   # UMAP dimensions

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

# Dynamically query all RDKit 2D Descriptor names
ALL_RDKIT_DESCRIPTORS = [d[0] for d in Descriptors._descList]

# Load the Math-Trimmed keep list generated from full_descriptor_audit.py
KEEP_LISTS_PATH = os.path.join(OUTPUT_DIR, "descriptor_keep_lists.json")
if os.path.exists(KEEP_LISTS_PATH):
    with open(KEEP_LISTS_PATH, "r") as f:
        MATH_TRIMMED_KEEP_LISTS = json.load(f)
    print(f" -> Successfully loaded math-trimmed keep lists from {KEEP_LISTS_PATH}")
else:
    raise FileNotFoundError(
        f"Could not find '{KEEP_LISTS_PATH}'. Please run full_descriptor_audit.py "
        "first to generate your data-driven descriptor keep lists."
    )

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_all_rdkit_2d_descriptors(mol):
    """Calculates all ~200 available RDKit 2D descriptors for a molecule."""
    if mol is None:
        return np.zeros(len(ALL_RDKIT_DESCRIPTORS), dtype=np.float64)
    
    values = []
    for name, func in Descriptors._descList:
        try:
            v = func(mol)
            values.append(v if np.isfinite(v) else 0.0)
        except Exception:
            values.append(0.0)
    return np.array(values, dtype=np.float64)


def create_scaffold_folds(mols, n_folds=5):
    """Groups molecules by Murcko Scaffold and distributes them evenly across folds."""
    scaffolds = {}
    for idx, mol in enumerate(mols):
        if mol is None: continue
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(core)
        except:
            scaffold_smiles = "" # Fallback

        if scaffold_smiles not in scaffolds:
            scaffolds[scaffold_smiles] = []
        scaffolds[scaffold_smiles].append(idx)

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


def decide_log_transform(y, category):
    skew = pd.Series(y).skew()
    if skew > SKEW_THRESHOLD and np.all(y >= 0):
        print(f" -> Target skew = {skew:.2f} (> {SKEW_THRESHOLD}) -> applying log1p transform")
        return np.log1p(y), True, skew
    print(f" -> Target skew = {skew:.2f} -> no transform applied")
    return y, False, skew


def detect_ceiling(y):
    y_max = y.max()
    frac_at_ceiling = np.mean(np.isclose(y, y_max, atol=CEILING_ATOL))
    has_ceiling = frac_at_ceiling >= CEILING_FRACTION_THRESHOLD
    if has_ceiling:
        print(f" -> WARNING: {frac_at_ceiling*100:.1f}% of values sit at max ({y_max:.2f}) "
              f"-> likely assay ceiling, will clip predictions at this value")
    return has_ceiling, y_max, frac_at_ceiling


# ==========================================
# MAIN PIPELINE
# ==========================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n=== INITIALIZING MASTER PIPELINE (MATH-TRIMMED FEATURE SET) ===")
    print(f" -> Output Directory: {OUTPUT_DIR}/")
    print(f" -> Fingerprint Radius: {FP_RADIUS}")
    print(f" -> Cross-Validation: Scaffold Split ({N_FOLDS} Folds)")

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=2048)
    results_summary = []

    for category, details in ENDPOINTS.items():
        print(f"\n{'='*50}")
        print(f"PROCESSING: {category.upper()}")
        print(f"{'='*50}")

        # 1. Load Data
        print(" -> Downloading dataset from TDC...")
        raw_df = details["class"](name=details["name"]).get_data()

        mols = [Chem.MolFromSmiles(s) for s in raw_df['Drug']]
        valid_idx = [i for i, m in enumerate(mols) if m is not None]
        mols = [mols[i] for i in valid_idx]
        y = raw_df['Y'].values[valid_idx].astype(np.float64)

        is_classification = len(np.unique(y)) <= 10
        print(f" -> Task Type: {'Classification' if is_classification else 'Regression'}")

        y_original = y.copy()
        log_transformed = False
        has_ceiling = False
        ceiling_value = None

        if not is_classification:
            y, log_transformed, skew = decide_log_transform(y, category)
            has_ceiling, ceiling_value, ceiling_frac = detect_ceiling(y_original)

        # 2. Extract Features (Fingerprints + Math-Trimmed Descriptors)
        print(" -> Extracting Morgan Fingerprints & UMAP embedding...")
        fp_list = []
        for mol in mols:
            fp = morgan_gen.GetFingerprint(mol)
            arr = np.zeros((0,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fp_list.append(arr)

        X_fp = np.stack(fp_list)
        reducer = umap.UMAP(n_components=FP_N_COMPONENTS, metric="jaccard", random_state=RANDOM_STATE)
        fp_embedding = reducer.fit_transform(X_fp)

        print(" -> Calculating complete RDKit 2D pool...")
        desc_matrix = np.stack([get_all_rdkit_2d_descriptors(m) for m in mols])
        
        # Clean potential NaNs safely
        desc_matrix = np.nan_to_num(desc_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Slice matrix to match only audited features found in the JSON file
        keep_names = MATH_TRIMMED_KEEP_LISTS.get(category, ALL_RDKIT_DESCRIPTORS)
        keep_indices = [i for i, name in enumerate(ALL_RDKIT_DESCRIPTORS) if name in keep_names]
        
        desc_filtered = desc_matrix[:, keep_indices]
        print(f" -> Filtered noise: utilizing {desc_filtered.shape[1]} descriptors based on math audit")

        desc_scaled = StandardScaler().fit_transform(desc_filtered)

        # Combine fixed UMAP components + scaled trimmed descriptors
        X_combined = np.hstack([fp_embedding, desc_scaled])

        # 3. Scaffold Splitting
        print(" -> Partitioning molecules by Murcko Scaffold...")
        scaffold_folds = create_scaffold_folds(mols, n_folds=N_FOLDS)

        # Save finalized feature matrices out to disk
        np.save(f"{OUTPUT_DIR}/{category}_X_trimmed.npy", X_combined)
        np.save(f"{OUTPUT_DIR}/{category}_y_trimmed.npy", y_original)
        np.save(f"{OUTPUT_DIR}/{category}_folds.npy", scaffold_folds)

        # 4. XGBoost Training & Evaluation
        print(" -> Training XGBoost via Scaffold CV...")

        fold_metric_a = []   
        fold_metric_b = []   
        fold_weights = []

        for test_fold in range(N_FOLDS):
            test_mask = (scaffold_folds == test_fold)
            train_mask = (scaffold_folds != test_fold)

            X_train, X_test = X_combined[train_mask], X_combined[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]                    
            y_test_original = y_original[test_mask]                          

            if len(y_test) == 0:
                continue

            if is_classification:
                if len(np.unique(y_test)) < 2:
                    print(f"    Fold {test_fold}: skipped (only one class present in test fold)")
                    continue

                model = xgb.XGBClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=6,
                    random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1
                )
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]

                roc_auc = roc_auc_score(y_test, probs)
                auprc = average_precision_score(y_test, probs)

                fold_metric_a.append(roc_auc)
                fold_metric_b.append(auprc)
                fold_weights.append(len(y_test))
                print(f"    Fold {test_fold}: ROC-AUC={roc_auc:.4f}  AUPRC={auprc:.4f}  (Size: {len(y_test)})")

            else:
                # Regularize Excretion specifically due to significant environmental noise
                if category == "Excretion":
                    model = xgb.XGBRegressor(
                        n_estimators=60, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=RANDOM_STATE, eval_metric='rmse', n_jobs=-1
                    )
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

                r2 = r2_score(y_test_original, preds)
                rmse = np.sqrt(mean_squared_error(y_test_original, preds))

                fold_metric_a.append(r2)
                fold_metric_b.append(rmse)
                fold_weights.append(len(y_test))
                print(f"    Fold {test_fold}: R2={r2:.4f}  RMSE={rmse:.4f}  (Size: {len(y_test)})")

        if len(fold_metric_a) == 0:
            print(f" -> WARNING: no valid folds for {category}, skipping summary")
            continue

        final_metric_a = np.average(fold_metric_a, weights=fold_weights)
        final_metric_b = np.average(fold_metric_b, weights=fold_weights)

        if is_classification:
            label_a, label_b = "ROC-AUC", "AUPRC"
        else:
            label_a, label_b = "R2", "RMSE"

        print(f"\n>>> FINAL {label_a} (Scaffold CV): {final_metric_a:.4f} <<<")
        print(f">>> FINAL {label_b} (Scaffold CV): {final_metric_b:.4f} <<<")

        results_summary.append({
            "Dataset": category,
            label_a: final_metric_a,
            label_b: final_metric_b,
            "LogTransformed": log_transformed if not is_classification else "-",
            "CeilingClipped": has_ceiling if not is_classification else "-",
            "FeaturesUsed": X_combined.shape[1]
        })

    # ==========================================
    # FINAL SUMMARY REPORT
    # ==========================================
    print("\n" + "="*60)
    print("ALL OPTIMIZED MODELS EVALUATED (SCAFFOLD SPLIT)")
    print("="*60)
    df_results = pd.DataFrame(results_summary)
    print(df_results.to_string(index=False))