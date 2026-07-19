import warnings
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tdc.single_pred import ADME, Tox

# 1. Clear terminal clutter
warnings.filterwarnings("ignore")

# Initialize the fingerprint machine
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_ecfp(smiles):
    """Converts a raw chemical SMILES string into a 2048-bit numerical fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(2048, dtype=np.int8)
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((0,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(2048, dtype=np.int8)

if __name__ == "__main__":
    print("=== STARTING HACKATHON DATA PIPELINE ===")
    
    endpoints = {
        "Absorption":  {"class": ADME, "name": "Caco2_Wang"},
        "Distribution":{"class": ADME, "name": "BBB_Martins"},
        "Metabolism":  {"class": ADME, "name": "CYP2C9_Veith"},
        "Excretion":   {"class": ADME, "name": "Clearance_Hepatocyte_AZ"},
        "Toxicity":    {"class": Tox,  "name": "hERG"}
    }
    
    # Create a clean directory folder to hold the saved data
    os.makedirs("processed_data", exist_ok=True)
    
    for category, details in endpoints.items():
        print(f"\nProcessing {category} ({details['name']})...")
        
        raw_df = details["class"](name=details["name"]).get_data()
        raw_df['features'] = raw_df['Drug'].apply(smiles_to_ecfp)
        
        X = np.stack(raw_df['features'].values)
        y = raw_df['Y'].values
        
        # --- NEW STEP: Save directly to your hard drive ---
        features_path = f"processed_data/{category}_X.npy"
        targets_path = f"processed_data/{category}_y.npy"
        
        np.save(features_path, X)
        np.save(targets_path, y)
        
        print(f" -> Saved to disk: {features_path} {X.shape}")
        print(f" -> Saved to disk: {targets_path} {y.shape}")

    print("\n=== SUCCESS: ALL DATA PERMANENTLY STORED ON HARD DRIVE! ===")