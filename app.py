import os
import json
import gradio as gr
import numpy as np
import umap
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
import xgboost as xgb

# Configurations
FP_RADIUS = 3
FP_N_COMPONENTS = 10
ALL_RDKIT_DESCRIPTORS = [d[0] for d in Descriptors._descList]

# Load Keep Lists
with open("scaffold_dataset/descriptor_keep_lists.json", "r") as f:
    KEEP_LISTS = json.load(f)

# Load Production Models, UMAPs, and Scalers
MODELS = {}
UMAPS = {}
SCALERS = {}
CATEGORIES = ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]

for cat in CATEGORIES:
    model_path = f"models/{cat}_model.json"
    umap_path = f"models/{cat}_umap.joblib"
    scaler_path = f"models/{cat}_scaler.joblib"
    
    if os.path.exists(model_path) and os.path.exists(umap_path) and os.path.exists(scaler_path):
        is_clf = cat in ["Distribution", "Metabolism", "Toxicity"]
        MODELS[cat] = xgb.XGBClassifier() if is_clf else xgb.XGBRegressor()
        MODELS[cat].load_model(model_path)
        UMAPS[cat] = joblib.load(umap_path)
        SCALERS[cat] = joblib.load(scaler_path)

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=2048)

def predict_admet(smiles_input):
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        return {"Error": "Invalid SMILES String code."}
    
    # 1. Structural Fingerprints (Raw)
    fp = morgan_gen.GetFingerprint(mol)
    fp_arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
    X_fp = fp_arr.reshape(1, -1)
    
    # 2. Compute Global 2D Descriptors (Raw)
    desc_vals = []
    for name, func in Descriptors._descList:
        try:
            v = func(mol)
            desc_vals.append(v if np.isfinite(v) else 0.0)
        except:
            desc_vals.append(0.0)
    desc_matrix = np.array(desc_vals).reshape(1, -1)
    desc_matrix = np.nan_to_num(desc_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    outputs = {}
    for cat in MODELS.keys():
        model = MODELS[cat]
        reducer = UMAPS[cat]
        scaler = SCALERS[cat]
        
        # Transform UMAP using the saved reducer
        fp_embedding = reducer.transform(X_fp)
        
        # Filter and Scale Descriptors using the saved scaler
        keep_names = KEEP_LISTS[cat]
        keep_indices = [i for i, name in enumerate(ALL_RDKIT_DESCRIPTORS) if name in keep_names]
        desc_filtered = desc_matrix[:, keep_indices]
        desc_scaled = scaler.transform(desc_filtered)
        
        # Combine UMAP components + Trimmed Scaled Descriptors
        X_input = np.hstack([fp_embedding, desc_scaled])
        
        # Predict
        if cat in ["Distribution", "Metabolism", "Toxicity"]:
            prob = model.predict_proba(X_input)[0][1]
            outputs[cat] = f"Positive (Prob: {prob:.2f})" if prob > 0.5 else f"Negative (Prob: {prob:.2f})"
        else:
            pred = model.predict(X_input)[0]
            if cat == "Excretion":
                pred = np.expm1(pred) # Inverse log1p transform
                pred = np.clip(pred, None, 150.0) # Apply assay ceiling clip
            outputs[cat] = f"{pred:.4f}"
            
    return outputs

# Build UI layout
demo = gr.Interface(
    fn=predict_admet,
    inputs=gr.Textbox(placeholder="Enter SMILES code here (e.g., CC(=O)NC1=CC=C(O)C=C1)...", label="Input Molecule"),
    outputs=gr.JSON(label="Predicted ADMET Metrics"),
    title="ADMET Properties Prediction Pipeline",
    description="Input a validated SMILES string structure to compute UMAP-Morgan embeddings combined with audited RDKit 2D descriptor selections."
)

if __name__ == "__main__":
    demo.launch()