import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def get_model_filename(model_name):
    """
    Returns the file name mapping for a given model name.
    
    Parameters:
    model_name (str): The readable name of the classifier.
    
    Returns:
    str: The file name of the pickled model check point.
    """
    mapping = {
        'Logistic Regression': 'model_logistic_regression.pkl',
        'Decision Tree': 'model_decision_tree.pkl',
        'Random Forest': 'model_random_forest.pkl',
        'XGBoost': 'model_xgboost.pkl'
    }
    # Fallback to logistic regression if model name is unrecognized
    return mapping.get(model_name, 'model_logistic_regression.pkl')

def run_pipeline(model_to_train=None):
    """
    Executes the complete machine learning pipeline:
    1. Loads dataset and validates schema constraints.
    2. Performs security-specific domain feature engineering.
    3. Splits features and target, and sets up train/test partitions.
    4. Configures data preprocessing workflows (Standard scaling & One-hot encoding).
    5. Defines candidates and trains model(s) based on input filters.
    6. Performs evaluation and prints leaderboard comparison.
    7. Extracts feature importances and writes model results/checkpoints.
    8. Scores the entire dataset to generate device threat predictions.
    
    Parameters:
    model_to_train (str, optional): Name of a specific model to train. If None, trains all models.
    """
    print("="*60)
    print("STARTING IoT DEVICE COMPROMISE DETECTION ML PIPELINE")
    print("="*60)
    
    # Define path to raw network security dataset
    csv_path = "it_security_iot_botnet_ddos_detection_80k.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing input dataset: {csv_path}")
        
    print(f"Loading dataset from: {csv_path}")
    # Load dataset using pandas
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Validate target presence in dataset
    target_col = 'Is_Compromised_Label'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
        
    print("\nEngineering security-specific features...")
    # Create copy to avoid mutating source dataframe
    df_processed = df.copy()
    
    # Feature 1: packets_per_ip (Measure density of packets per target IP - flags high packet count targeting few IPs)
    df_processed['packets_per_ip'] = df_processed['Outbound_Packets_Sec'] / (df_processed['Unique_Dest_IPs'] + 1)
    
    # Feature 2: cpu_intensity_per_packet (CPU cycle consumption per packet sent - signals anomalous CPU-bound malware)
    df_processed['cpu_intensity_per_packet'] = df_processed['Device_CPU_Usage_Pct'] / (df_processed['Outbound_Packets_Sec'] + 1)
    
    # Feature 3: network_bandwidth_est (Calculated payload rate: packet frequency * average size in bytes)
    df_processed['network_bandwidth_est'] = df_processed['Outbound_Packets_Sec'] * df_processed['Avg_Packet_Size_Bytes']
    
    # Feature 4: high_risk_device (Binary threshold flag indicating if calculated botnet likelihood exceeds threshold of 50)
    df_processed['high_risk_device'] = (df_processed['Calculated_Botnet_Score'] > 50).astype(int)
    
    print("  Feature engineering complete!")
    print("  New features created: packets_per_ip, cpu_intensity_per_packet, network_bandwidth_est, high_risk_device")
    
    # Drop telemetry labels and identifier columns not used in classification
    X = df_processed.drop(columns=['Device_ID', target_col])
    y = df_processed[target_col]
    
    # Segment columns into numeric and categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"  Numerical features: {len(numerical_cols)} -> {numerical_cols}")
    print(f"  Categorical features: {len(categorical_cols)} -> {categorical_cols}")
    
    # Perform stratified split to maintain device compromise class balances (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Split complete: Train={X_train.shape[0]:,}, Test={X_test.shape[0]:,}")
    
    # Define ColumnTransformer to apply preprocessing steps depending on column types:
    # - Standardize numeric values using StandardScaler
    # - One-Hot Encode string categories using OneHotEncoder (handling unseen testing categories gracefully)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Define dictionary of classification algorithms to assess
    all_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Filter list if user requested to train a specific model checkpoint
    if model_to_train and model_to_train in all_models:
        print(f"Filtering pipeline to train only: {model_to_train}")
        models = {model_to_train: all_models[model_to_train]}
    else:
        models = all_models
    
    results = []
    trained_pipelines = {}
    
    print("\nTraining and evaluating candidate models...")
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        # Create pipeline enclosing preprocessing steps and model logic
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Fit pipeline on target training slice
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline
        
        # Predict binary classifications and continuous probability logs
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate standard performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Append stats for pipeline evaluation logging
        results.append({
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        print(f"    ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        # Save trained pipeline checkpoint for the specific model using joblib
        specific_filename = get_model_filename(model_name)
        joblib.dump(pipeline, specific_filename)
        print(f"    Saved checkpoint to: {specific_filename}")
        
    # Sort performance leaderboard by ROC-AUC metric in descending order
    df_results = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
    print("\nModel Leaderboard:")
    print(df_results.to_string(index=False))
    
    # Save the leaderboard metrics to CSV format
    df_results.to_csv('model_results.csv', index=False)
    print("  Model results saved to: model_results.csv")
    
    # The highest-performing model is positioned at the top of sorted dataframe
    champion_name = df_results.loc[0, 'Model']
    champion_pipeline = trained_pipelines[champion_name]
    print(f"\nCHAMPION MODEL SELECTED: {champion_name} ({df_results.loc[0, 'ROC-AUC']:.4f} ROC-AUC)")
    
    # Save the general champion model to default filename champion_model.pkl
    model_file = 'champion_model.pkl'
    joblib.dump(champion_pipeline, model_file)
    print(f"  Champion model saved to: {model_file}")
    
    print("\nExtracting feature importances...")
    try:
        # Resolve best model to extract feature importances
        if 'Random Forest' in trained_pipelines:
            best_model_for_importance = trained_pipelines['Random Forest'].named_steps['classifier']
            preprocessor_obj = trained_pipelines['Random Forest'].named_steps['preprocessor']
        elif 'XGBoost' in trained_pipelines:
            best_model_for_importance = trained_pipelines['XGBoost'].named_steps['classifier']
            preprocessor_obj = trained_pipelines['XGBoost'].named_steps['preprocessor']
        else:
            best_model_for_importance = champion_pipeline.named_steps['classifier']
            preprocessor_obj = champion_pipeline.named_steps['preprocessor']
            
        # Reconstruct high-dimensional feature names resulting from one-hot encoding
        cat_encoder = preprocessor_obj.named_transformers_['cat']
        cat_features_encoded = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        all_features = numerical_cols + cat_features_encoded
        
        # Access model parameter attributes (importances vs linear coefficients)
        if hasattr(best_model_for_importance, 'feature_importances_'):
            importances = best_model_for_importance.feature_importances_
        elif hasattr(best_model_for_importance, 'coef_'):
            importances = np.abs(best_model_for_importance.coef_[0])
        else:
            importances = np.ones(len(all_features)) / len(all_features)
            
        # Sort and write feature importances list to CSV
        indices = np.argsort(importances)[::-1]
        df_importance = pd.DataFrame({
            'Feature': [all_features[i] for i in indices],
            'Importance': importances[indices]
        })
        df_importance.to_csv('feature_importance.csv', index=False)
        print("  Feature importances saved to: feature_importance.csv")
    except Exception as e:
        print(f"  Warning: Could not extract feature importances: {e}")
        
    print("\nGenerating threat scores for entire dataset...")
    # Clean input feature matrix matching prediction configuration
    full_X = df_processed.drop(columns=['Device_ID', target_col])
    # Compute output probabilities using selected champion model
    threat_scores = champion_pipeline.predict_proba(full_X)[:, 1]
    predictions = champion_pipeline.predict(full_X)
    
    # Store scores and predictions along with device IDs
    scored_df = pd.DataFrame({
        'Device_ID': df['Device_ID'],
        'threat_score': threat_scores,
        'predicted_compromise': predictions
    })
    
    # Bin continuous threat scores into actionable security levels consistently:
    # - Low Threat: score < 0.30
    # - Medium Threat: 0.30 <= score < 0.70
    # - High Threat: score >= 0.70
    scored_df['threat_category'] = pd.cut(
        scored_df['threat_score'],
        bins=[-0.0001, 0.3, 0.7, 1.0001],
        labels=['Low Threat', 'Medium Threat', 'High Threat'],
        right=False
    )
    
    # Save processed threat output database
    scored_df.to_csv('threat_scores.csv', index=False)
    print("  Threat scores saved to: threat_scores.csv")
    
    print("\n" + "="*60)
    print("ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
