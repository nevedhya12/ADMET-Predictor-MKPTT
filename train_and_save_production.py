import os
import json
import numpy as np
import xgboost as xgb
import umap
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from tdc.single_pred import ADME, Tox

OUTPUT_DIR = "scaffold_dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FP_RADIUS = 3
FP_N_COMPONENTS = 10
ALL_RDKIT_DESCRIPTORS = [d[0] for d in Descriptors._descList]

# Load Keep Lists
with open(f"{OUTPUT_DIR}/descriptor_keep_lists.json", "r") as f:
    KEEP_LISTS = json.load(f)

ENDPOINTS = {
    "Absorption":   {"class": ADME, "name": "Caco2_Wang"},
    "Distribution": {"class": ADME, "name": "BBB_Martins"},
    "Metabolism":   {"class": ADME, "name": "CYP2C9_Veith"},
    "Excretion":    {"class": ADME, "name": "Clearance_Hepatocyte_AZ"},
    "Toxicity":     {"class": Tox,  "name": "hERG"},
}

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=2048)

for cat, details in ENDPOINTS.items():
    print(f"Re-training and saving objects for: {cat}...")
    
    # 1. Load Raw Data
    raw_df = details["class"](name=details["name"]).get_data()
    mols = [Chem.MolFromSmiles(s) for s in raw_df['Drug']]
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in valid_idx]
    y = raw_df['Y'].values[valid_idx].astype(np.float64)
    
    # 2. Log transform Excretion
    if cat == "Excretion":
        y = np.log1p(y)
        
    # 3. Compute Fingerprints
    fp_list = []
    for mol in mols:
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((0,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr)
    X_fp = np.stack(fp_list)
    
    # Fit & Save UMAP
    reducer = umap.UMAP(n_components=FP_N_COMPONENTS, metric="jaccard", random_state=42)
    fp_embedding = reducer.fit_transform(X_fp)
    joblib.dump(reducer, f"{MODEL_DIR}/{cat}_umap.joblib")
    
    # 4. Compute Descriptors
    desc_matrix = []
    for mol in mols:
        vals = []
        for name, func in Descriptors._descList:
            try:
                v = func(mol)
                vals.append(v if np.isfinite(v) else 0.0)
            except:
                vals.append(0.0)
        desc_matrix.append(vals)
    desc_matrix = np.array(desc_matrix)
    desc_matrix = np.nan_to_num(desc_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    keep_names = KEEP_LISTS.get(cat, ALL_RDKIT_DESCRIPTORS)
    keep_indices = [i for i, name in enumerate(ALL_RDKIT_DESCRIPTORS) if name in keep_names]
    desc_filtered = desc_matrix[:, keep_indices]
    
    # Fit & Save Scaler
    scaler = StandardScaler()
    desc_scaled = scaler.fit_transform(desc_filtered)
    joblib.dump(scaler, f"{MODEL_DIR}/{cat}_scaler.joblib")
    
    # 5. Combine & Train
    X = np.hstack([fp_embedding, desc_scaled])
    
    is_clf = len(np.unique(y)) <= 10
    if is_clf:
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    else:
        if cat == "Excretion":
            model = xgb.XGBRegressor(n_estimators=60, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
        else:
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    
    model.fit(X, y)
    model.save_model(f"{MODEL_DIR}/{cat}_model.json")
    print(f" -> Saved Model, UMAP, and Scaler for {cat} successfully.")