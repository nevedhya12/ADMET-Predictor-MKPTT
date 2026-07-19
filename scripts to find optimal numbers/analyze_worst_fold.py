import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score

def analyze_cluster_distances(category="Absorption"):
    print(f"=== Extrapolation Analysis for {category} ===")
    
    # 1. Load the processed arrays
    X = np.load(f"processed_data/{category}_combined_X.npy")
    y = np.load(f"processed_data/{category}_y.npy")
    clusters = np.load(f"processed_data/{category}_clusters.npy")
    
    unique_clusters = np.unique(clusters)
    is_classification = len(np.unique(y)) <= 10
    
    # 2. Find the worst-performing fold programmatically
    worst_score = float('inf') if not is_classification else 1.0
    worst_cluster = -1
    
    print(" -> Identifying the worst fold...")
    for test_cluster in unique_clusters:
        test_mask = (clusters == test_cluster)
        train_mask = (clusters != test_cluster)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        if is_classification:
            model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss', n_jobs=-1)
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            # Lower accuracy is worse
            if score < worst_score:
                worst_score = score
                worst_cluster = test_cluster
        else:
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='rmse', n_jobs=-1)
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))
            # Lower R2 is worse
            if score < worst_score:
                worst_score = score
                worst_cluster = test_cluster

    metric_name = "Accuracy" if is_classification else "R2 Score"
    print(f" -> Worst Fold: Cluster {worst_cluster} ({metric_name}: {worst_score:.4f})")

    # 3. Calculate the centroid for every cluster
    centroids = {}
    for c in unique_clusters:
        centroids[c] = np.mean(X[clusters == c], axis=0)

    # 4. Calculate distances from the worst fold to all others
    worst_centroid = centroids[worst_cluster]
    distances_from_worst = []
    
    print(f"\nDistances from Worst Fold (Cluster {worst_cluster}) to others:")
    for c in unique_clusters:
        if c != worst_cluster:
            dist = np.linalg.norm(worst_centroid - centroids[c])
            distances_from_worst.append(dist)
            print(f"    -> To Cluster {c}: {dist:.4f}")
            
    avg_dist_from_worst = np.mean(distances_from_worst)
    print(f"\nAverage distance from Worst Fold to others: {avg_dist_from_worst:.4f}")

    # 5. Establish a baseline: Average distance between all OTHER pairs
    other_clusters = [c for c in unique_clusters if c != worst_cluster]
    other_pairwise_distances = []
    for i in range(len(other_clusters)):
        for j in range(i + 1, len(other_clusters)):
            dist = np.linalg.norm(centroids[other_clusters[i]] - centroids[other_clusters[j]])
            other_pairwise_distances.append(dist)
            
    avg_dist_others = np.mean(other_pairwise_distances)
    print(f"Average distance between all OTHER clusters: {avg_dist_others:.4f}")

    # 6. Conclusion and Outlier Ratio
    ratio = avg_dist_from_worst / avg_dist_others
    print("-" * 50)
    print(f"Outlier Ratio (Worst vs Baseline): {ratio:.2f}x")
    
    if ratio > 1.3:
        print("Verdict: STRONG EVIDENCE of an extrapolation penalty.")
        print("The worst fold is structurally isolated in the combined feature space.")
    else:
        print("Verdict: WEAK EVIDENCE of an extrapolation penalty.")
        print("The worst fold is NOT uniquely isolated. The poor performance is likely driven by target variance, label noise, or a complex localized decision boundary rather than pure distance.")
    print("-" * 50)

if __name__ == "__main__":
    analyze_cluster_distances("Absorption")