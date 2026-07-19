import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score
import warnings

# Mute warnings for clean terminal output
warnings.filterwarnings("ignore")

def train_spatial_cv(category):
    print(f"\n=== Training XGBoost for {category} ===")
    
    # 1. Load the processed arrays and cluster assignments
    X = np.load(f"processed_data/{category}_combined_X.npy")
    y = np.load(f"processed_data/{category}_y.npy")
    clusters = np.load(f"processed_data/{category}_clusters.npy")
    
    unique_clusters = np.unique(clusters)
    
    # 2. DYNAMIC TASK DETECTION
    # If there are 10 or fewer unique answers (like 0 and 1), it is a multiple-choice Classification
    # If there are hundreds of unique decimals, it is a Continuous Regression
    is_classification = len(np.unique(y)) <= 10
    
    task_type = "Classification" if is_classification else "Regression"
    print(f" -> Detected Task Type: {task_type}")
    
    fold_scores = []
    fold_weights = []
    
    # 3. Loop through each of the 5 chemical families
    for test_cluster in unique_clusters:
        # Create boolean masks to separate the test island from the training islands
        test_mask = (clusters == test_cluster)
        train_mask = (clusters != test_cluster)
        
        # Slice the data
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        fold_size = len(y_test)
        
        # 4. Initialize and Train the Correct Model Type
        if is_classification:
            model = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, 
                random_state=42, eval_metric='logloss', n_jobs=-1
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Score classification using standard Accuracy
            score = accuracy_score(y_test, predictions)
            metric_name = "Accuracy"
            
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, 
                random_state=42, eval_metric='rmse', n_jobs=-1
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Score regression using R-squared 
            score = r2_score(y_test, predictions)
            metric_name = "R2 Score"
            
        print(f" -> Fold {test_cluster} (Size: {fold_size:4d}): {metric_name} = {score:.4f}")
        
        fold_scores.append(score)
        fold_weights.append(fold_size)
        
    # 5. Calculate the final metrics
    standard_score = np.mean(fold_scores)
    weighted_score = np.average(fold_scores, weights=fold_weights)
    
    print("-" * 50)
    print(f"Standard Average {metric_name} (Flawed):   {standard_score:.4f}")
    print(f"Weighted Average {metric_name} (Robust):   {weighted_score:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    # List of all 5 chemical property blocks saved on your drive
    all_categories = ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]
    
    # Run the rigorous spatial cross-validation loop for every dataset
    for current_category in all_categories:
        train_spatial_cv(current_category)
        
    print("\n=== ALL MODELS SUCCESSFULLY TRAINED AND EVALUATED! ===")