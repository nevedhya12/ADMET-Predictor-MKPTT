import numpy as np
import xgboost as xgb
import warnings

# Mute warnings for clean terminal output
warnings.filterwarnings("ignore")

# Your exact 15 descriptors in the exact order they were concatenated
DESCRIPTOR_NAMES = [
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
    "AromaticRings", "FormalCharge", "MolarRefractivity", "Fsp3", "QED",
    "AmideBondCount", "MaxRingSize", "BertzCT", "StereocenterCount",
]

# The number of UMAP dimensions that come BEFORE the descriptors in X_combined
FP_N_COMPONENTS = 10 

def audit_descriptors(category):
    print(f"\n{'='*50}")
    print(f"=== Auditing Descriptors for: {category} ===")
    
    # 1. Load the combined dataset
    X = np.load(f"processed_data/{category}_combined_X.npy")
    y = np.load(f"processed_data/{category}_y.npy")
    
    # 2. Dynamic Task Detection (same as your training loop)
    is_classification = len(np.unique(y)) <= 10
    
    if is_classification:
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        # Apply the same log transform logic for continuous positive targets (Excretion)
        if np.all(y > 0):
            y = np.log(y)
            
    # 3. Train the model on the full dataset to get global importances
    model.fit(X, y)
    
    # 4. Extract Information Gain scores (they automatically sum to 1.0 or 100%)
    importances = model.feature_importances_
    
    # Slice the array to ONLY look at the 15 descriptors (ignoring the 10 UMAP columns)
    descriptor_importances = importances[FP_N_COMPONENTS:]
    
    # Pair the names with their scores and sort them from worst to best
    desc_imp_pairs = list(zip(DESCRIPTOR_NAMES, descriptor_importances))
    desc_imp_pairs.sort(key=lambda x: x[1])
    
    # 5. Report the Noise
    useless_desc = []
    print("Feature Importance Scores:")
    
    for name, score in desc_imp_pairs:
        # Convert raw decimal to a clean percentage
        percent_importance = score * 100 
        
        # If it contributes less than 1%, it is mathematical noise
        if percent_importance < 1.0:
            useless_desc.append((name, percent_importance))
        else:
            print(f" [+] {name:20s}: {percent_importance:>5.2f}%")
            
    print("-" * 50)
    if useless_desc:
        print("🚨 NOISE DETECTED! Drop these for this specific dataset:")
        for name, score in useless_desc:
            print(f" [-] {name:20s}: {score:>5.2f}% importance")
    else:
        print("✅ Clean Data! All descriptors are contributing meaningfully above 1%.")

if __name__ == "__main__":
    all_categories = ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]
    
    for current_category in all_categories:
        audit_descriptors(current_category)
        
    print(f"\n{'='*50}")
    print("AUDIT COMPLETE.")