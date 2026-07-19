from tdc.single_pred import ADME, Tox
import pandas as pd

print("Downloading Datasets...")

# 1. A - Absorption: Caco-2 Cell Permeability (Regression)
caco2_df = ADME(name='Caco2_Wang').get_data()
print(f"Caco-2 dataset loaded: {len(caco2_df)} molecules")

# 2. D - Distribution: Blood-Brain Barrier (Classification)
bbb_df = ADME(name='BBB_Martins').get_data()
print(f"BBB dataset loaded: {len(bbb_df)} molecules")

# 3. M - Metabolism: CYP2C9 Inhibition (Classification)
cyp2c9_df = ADME(name='CYP2C9_Veith').get_data()
print(f"CYP2C9 dataset loaded: {len(cyp2c9_df)} molecules")

# 4. E - Excretion: Clearance Rate (Regression)
clearance_df = ADME(name='Clearance_Hepatocyte_AZ').get_data()
print(f"Clearance dataset loaded: {len(clearance_df)} molecules")

# 5. T - Toxicity: hERG Cardiotoxicity (Classification)
herg_df = Tox(name='hERG').get_data()
print(f"hERG dataset loaded: {len(herg_df)} molecules")

print("\n--- Example Data (Blood-Brain Barrier) ---")
print(caco2_df.head())