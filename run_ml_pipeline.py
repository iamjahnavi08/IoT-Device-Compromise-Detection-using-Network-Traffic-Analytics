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
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def run_pipeline():
    print("="*60)
    print("STARTING IoT DEVICE COMPROMISE DETECTION ML PIPELINE")
    print("="*60)
    
    csv_path = "it_security_iot_botnet_ddos_detection_80k.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing input dataset: {csv_path}")
        
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    target_col = 'Is_Compromised_Label'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
        
    print("\nEngineering security-specific features...")
    df_processed = df.copy()
    
    df_processed['packets_per_ip'] = df_processed['Outbound_Packets_Sec'] / (df_processed['Unique_Dest_IPs'] + 1)
    df_processed['cpu_intensity_per_packet'] = df_processed['Device_CPU_Usage_Pct'] / (df_processed['Outbound_Packets_Sec'] + 1)
    df_processed['network_bandwidth_est'] = df_processed['Outbound_Packets_Sec'] * df_processed['Avg_Packet_Size_Bytes']
    df_processed['high_risk_device'] = (df_processed['Calculated_Botnet_Score'] > 50).astype(int)
    
    print("  Feature engineering complete!")
    print("  New features created: packets_per_ip, cpu_intensity_per_packet, network_bandwidth_est, high_risk_device")
    
    X = df_processed.drop(columns=['Device_ID', target_col])
    y = df_processed[target_col]
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"  Numerical features: {len(numerical_cols)} -> {numerical_cols}")
    print(f"  Categorical features: {len(categorical_cols)} -> {categorical_cols}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Split complete: Train={X_train.shape[0]:,}, Test={X_test.shape[0]:,}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
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
    
    results = []
    trained_pipelines = {}
    
    print("\nTraining and evaluating candidate models...")
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        print(f"    ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
    df_results = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
    print("\nModel Leaderboard:")
    print(df_results.to_string(index=False))
    
    champion_name = df_results.loc[0, 'Model']
    champion_pipeline = trained_pipelines[champion_name]
    print(f"\nCHAMPION MODEL SELECTED: {champion_name} ({df_results.loc[0, 'ROC-AUC']:.4f} ROC-AUC)")
    
    print("\nExtracting feature importances...")
    try:
        if 'Random Forest' in trained_pipelines:
            best_model_for_importance = trained_pipelines['Random Forest'].named_steps['classifier']
            preprocessor_obj = trained_pipelines['Random Forest'].named_steps['preprocessor']
        else:
            best_model_for_importance = champion_pipeline.named_steps['classifier']
            preprocessor_obj = champion_pipeline.named_steps['preprocessor']
            
        cat_encoder = preprocessor_obj.named_transformers_['cat']
        cat_features_encoded = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        all_features = numerical_cols + cat_features_encoded
        
        if hasattr(best_model_for_importance, 'feature_importances_'):
            importances = best_model_for_importance.feature_importances_
        elif hasattr(best_model_for_importance, 'coef_'):
            importances = np.abs(best_model_for_importance.coef_[0])
        else:
            importances = np.ones(len(all_features)) / len(all_features)
            
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
    full_X = df_processed.drop(columns=['Device_ID', target_col])
    threat_scores = champion_pipeline.predict_proba(full_X)[:, 1]
    predictions = champion_pipeline.predict(full_X)
    
    scored_df = pd.DataFrame({
        'Device_ID': df['Device_ID'],
        'threat_score': threat_scores,
        'predicted_compromise': predictions
    })
    
    scored_df['threat_category'] = pd.cut(
        scored_df['threat_score'],
        bins=[0.0, 0.3, 0.7, 1.0],
        labels=['Low Threat', 'Medium Threat', 'High Threat'],
        include_lowest=True
    )
    
    scored_df.to_csv('threat_scores.csv', index=False)
    print("  Threat scores saved to: threat_scores.csv")
    
    model_file = 'champion_model.pkl'
    joblib.dump(champion_pipeline, model_file)
    print(f"  Champion model saved to: {model_file}")
    
    df_results.to_csv('model_results.csv', index=False)
    print("  Model results saved to: model_results.csv")
    
    print("\n" + "="*60)
    print("ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
