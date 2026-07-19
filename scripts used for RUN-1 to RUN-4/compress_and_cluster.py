import os
import numpy as np
import umap
from sklearn.cluster import KMeans

def process_spatial_splits(category):
    print(f"\n--- Structuring Leak-Proof Splits for {category} ---")
    
    # 1. Load the raw matrix blocks from your hard drive
    X = np.load(f"processed_data/{category}_X.npy")
    y = np.load(f"processed_data/{category}_y.npy")
    print(f"Loaded {category} features matrix: {X.shape}")
    
    # 2. Compress the 2048 dimensions down to a 10D structural landscape
    print("Compressing chemical space to 10D coordinates via UMAP...")
    reducer = umap.UMAP(n_components=10, metric='jaccard', random_state=42)
    X_10d = reducer.fit_transform(X)
    
    # 3. Cluster the 10D space into 5 distinct chemical families
    print("Grouping molecules into structural cluster families...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_groups = kmeans.fit_predict(X_10d)
    
    # 4. EXPLICITLY SAVE THE CLUSTERS TO THE HARD DRIVE
    clusters_path = f"processed_data/{category}_clusters.npy"
    np.save(clusters_path, cluster_groups)
    print(f" -> SUCCESS: Saved cluster folds to disk at {clusters_path}")
    
    # Print out cluster sizes to terminal
    unique, counts = np.unique(cluster_groups, return_counts=True)
    print("Cluster sizes (Molecules per chemical family):")
    for cluster_id, count in zip(unique, counts):
        print(f" -> Cluster Family {cluster_id}: {count} molecules")
        
    return X, y, cluster_groups

if __name__ == "__main__":
    # List of all 5 chemical property blocks saved on your drive
    all_categories = ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]
    
    # Run the processing loop sequentially
    for current_category in all_categories:
        process_spatial_splits(current_category)
        print("-" * 50)
        
    print("\n=== ALL DATASETS COMPRESSED (10D), CLUSTERED, AND SAVED! ===")