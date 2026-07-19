"""
add_descriptors.py
(Upgraded with Dynamic Noise Filtering)
"""

import os
import warnings
import numpy as np
import umap
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, rdMolDescriptors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tdc.single_pred import ADME, Tox

warnings.filterwarnings("ignore")

DATA_DIR = "processed_data"
RANDOM_STATE = 42
FP_N_COMPONENTS = 10 

ENDPOINTS = {
    "Absorption":   {"class": ADME, "name": "Caco2_Wang", "k_clusters": 4},
    "Distribution": {"class": ADME, "name": "BBB_Martins", "k_clusters": 5},
    "Metabolism":   {"class": ADME, "name": "CYP2C9_Veith", "k_clusters": 4},
    "Excretion":    {"class": ADME, "name": "Clearance_Hepatocyte_AZ", "k_clusters": 5},
    "Toxicity":     {"class": Tox,  "name": "hERG", "k_clusters": 5},
}

DESCRIPTOR_NAMES = [
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
    "AromaticRings", "FormalCharge", "MolarRefractivity", "Fsp3", "QED",
    "AmideBondCount", "MaxRingSize", "BertzCT", "StereocenterCount",
]

# --- NEW: THE NOISE FILTER DICTIONARY ---
# Based exactly on your XGBoost Information Gain audit
NOISE_TO_DROP = {
    "Absorption": ["FormalCharge", "RotBonds", "HBA", "QED", "BertzCT", "AromaticRings"],
    "Distribution": ["FormalCharge"],
    "Metabolism": [],
    "Excretion": ["FormalCharge"],
    "Toxicity": []
}

_AMIDE_PATTERN = Chem.MolFromSmarts("C(=O)N")

def smiles_to_descriptors(smiles):
    """Returns a 15-element float vector, or zeros if the SMILES fails to parse."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float64)

        amide_count = len(mol.GetSubstructMatches(_AMIDE_PATTERN))
        ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
        max_ring_size = max(ring_sizes) if ring_sizes else 0
        stereo_count = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        ))

        return np.array([
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            Chem.GetFormalCharge(mol),
            Crippen.MolMR(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.qed(mol),
            amide_count,
            max_ring_size,
            GraphDescriptors.BertzCT(mol),
            stereo_count,
        ], dtype=np.float64)
    except Exception:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float64)


def process_category(category, details):
    print(f"\n{'=' * 60}")
    print(f"Category: {category}")
    print(f"{'=' * 60}")

    print(" -> Refetching TDC dataset to recover SMILES (same order as X.npy)...")
    raw_df = details["class"](name=details["name"]).get_data()
    smiles_list = raw_df["Drug"].values

    X_fp = np.load(f"{DATA_DIR}/{category}_X.npy")
    y = np.load(f"{DATA_DIR}/{category}_y.npy")
    if len(smiles_list) != X_fp.shape[0]:
        raise ValueError("Row count mismatch. Do not proceed.")

    print(" -> Computing RDKit descriptors...")
    descriptors = np.stack([smiles_to_descriptors(s) for s in smiles_list])
    
    # --- NEW: APPLYING THE NOISE FILTER ---
    drop_list = NOISE_TO_DROP.get(category, [])
    if drop_list:
        print(f" -> 🚨 Filtering out {len(drop_list)} noisy descriptors: {drop_list}")
        # Find the column index for every descriptor we want to KEEP
        keep_indices = [i for i, name in enumerate(DESCRIPTOR_NAMES) if name not in drop_list]
        # Slice the matrix to keep only the good columns
        descriptors = descriptors[:, keep_indices]
    else:
        print(" -> ✅ No noise detected. Keeping all 15 descriptors.")

    print(f" -> Running UMAP on fingerprint (n_components={FP_N_COMPONENTS})...")
    fp_reducer = umap.UMAP(n_components=FP_N_COMPONENTS, metric="jaccard", random_state=RANDOM_STATE)
    fp_embedding = fp_reducer.fit_transform(X_fp)

    print(" -> Scaling descriptors and concatenating with fingerprint embedding...")
    scaler = StandardScaler()
    descriptors_scaled = scaler.fit_transform(descriptors)
    X_combined = np.hstack([fp_embedding, descriptors_scaled])
    print(f" -> Combined feature shape: {X_combined.shape}")

    target_k = details["k_clusters"]
    print(f" -> Re-running KMeans({target_k}) on combined feature space...")
    kmeans = KMeans(n_clusters=target_k, random_state=RANDOM_STATE, n_init=10)
    cluster_groups = kmeans.fit_predict(X_combined)

    combined_path = f"{DATA_DIR}/{category}_combined_X.npy"
    clusters_path = f"{DATA_DIR}/{category}_clusters.npy"

    np.save(combined_path, X_combined)
    np.save(clusters_path, cluster_groups)

    print(f" -> Saved combined features -> {combined_path}")
    print(f" -> Saved (overwritten) clusters -> {clusters_path}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    for category, details in ENDPOINTS.items():
        process_category(category, details)

    print(f"\n{'=' * 60}")
    print("DONE. All features filtered, scaled, and clustered.")